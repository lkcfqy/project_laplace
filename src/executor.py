# src/executor.py
"""
Executor Module
--------------
This module handles the safe execution of generated code using a local sandbox.
It includes AST-based safety checks and timeout management.
"""
import subprocess
import sys
import tempfile
import os
try:
    try:
        from .docker_sandbox import DockerSandbox
    except ImportError:
        from docker_sandbox import DockerSandbox
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

import ast

class SafetyChecker(ast.NodeVisitor):
    def __init__(self):
        self.dangerous_imports = {'os', 'sys', 'subprocess', 'shutil', 'pickle', 'importlib', 'requests'}
        self.dangerous_functions = {'open', 'exec', 'eval', 'compile', 'input'}
        self.errors = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split('.')[0] in self.dangerous_imports:
                self.errors.append(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.split('.')[0] in self.dangerous_imports:
             self.errors.append(f"Forbidden import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.dangerous_functions:
            self.errors.append(f"Forbidden function call: {node.func.id}")
        self.generic_visit(node)

    def check(self, code_str):
        self.errors = []
        try:
            tree = ast.parse(code_str)
            self.visit(tree)
        except SyntaxError as e:
            return False, f"Syntax Error: {e}"
        
        if self.errors:
            return False, "\n".join(self.errors)
        return True, ""

class LocalSandbox:
    def __init__(self, timeout=5, use_docker=True):
        self.timeout = timeout
        self.safety = SafetyChecker()
        self.use_docker = use_docker and HAS_DOCKER
        if self.use_docker:
            try:
                self.docker_sandbox = DockerSandbox(timeout=timeout)
                print("🐳 Docker Sandbox activated.")
            except Exception as e:
                print(f"⚠️ Failed to init Docker Sandbox: {e}. Falling back to Local.")
                self.use_docker = False

    def execute(self, code_str: str, test_input=None):
        if self.use_docker:
            return self.docker_sandbox.execute(code_str, test_input)
            
        """
        验证安全性后，将生成的代码写入临时文件并运行。
        """
        # 1. 安全检查
        is_safe, error_msg = self.safety.check(code_str)
        if not is_safe:
            if "Syntax Error" in error_msg:
                 return False, "", f"Syntax Error in Generated Code:\n{error_msg}"
            return False, "", f"Security Violation:\n{error_msg}"

        # 2. 包装代码：添加测试调用的逻辑
        # 如果有 test_input，我们需要在代码末尾添加 print(solve(input))
        full_code = code_str
        # 再次确认 import os 不会被绕过 (AST 已查，这里是双保险或逻辑需要)
        # Note: full_code += ... 只是添加调用，不引入 import
        
        if test_input is not None:
             full_code += f"\n\nprint(solve({test_input}))"
        
        # 创建临时 Python 文件
        # 使用 delete=False 必须手动清理，这在 finally 块中处理
        # 优化：使用 try-finally 确保清理
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(full_code)
                tmp_path = tmp_file.name

            # 使用子进程运行代码
            # 注意：仍然存在风险（如通过 getattr 绕过 AST），但对于良性 Agent 足够了
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            
            output = result.stdout.strip()
            error = result.stderr.strip()
            
            success = (result.returncode == 0)
            return success, output, error

        except subprocess.TimeoutExpired:
            return False, "", "Execution Timed Out"
        except Exception as e:
            return False, "", str(e)
        finally:
            # 清理临时文件
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except: pass