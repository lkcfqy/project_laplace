import docker
import tarfile
import io
import time

class DockerSandbox:
    def __init__(self, image="python:3.10-slim", timeout=5, mem_limit="512m"):
        self.image = image
        self.timeout = timeout
        self.mem_limit = mem_limit
        self.client = docker.from_env()
        
        # Ensure image exists
        try:
            self.client.images.get(image)
        except docker.errors.ImageNotFound:
            print(f"🐳 Pulling Docker image: {image}...")
            self.client.images.pull(image)

    def execute(self, code_str: str, test_input=None):
        """
        Execute code inside a Docker container.
        """
        # Prepare code
        full_code = code_str
        if test_input is not None:
             full_code += f"\n\nprint(solve({test_input}))"
        
        # Create container
        container = None
        try:
            container = self.client.containers.run(
                self.image,
                command="python main.py",
                detach=True,
                mem_limit=self.mem_limit,
                network_mode="none", # Internet isolation
                # We need to inject the code. We'll do it via a tar archive or simply writing to stdin if possible,
                # but 'python -c' with long code is tricky. 
                # Better approach: sleep infinity, copy file, exec, kill.
                entrypoint="/bin/sh",
                tty=True,
            )
            
            # Copy code to container
            # Docker SDK put_archive expects a tar stream
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                data = full_code.encode('utf-8')
                info = tarfile.TarInfo(name='main.py')
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            tar_stream.seek(0)
            
            container.put_archive('/tmp', tar_stream) 
            
            # Execute
            # Exec run
            exec_res = container.exec_run(
                cmd="python /tmp/main.py",
                workdir="/tmp",
                demux=True # Separate stdout/stderr
            )
            
            stdout = exec_res.output[0]
            stderr = exec_res.output[1]
            
            output = stdout.decode('utf-8').strip() if stdout else ""
            error = stderr.decode('utf-8').strip() if stderr else ""
            
            success = (exec_res.exit_code == 0)
            
            return success, output, error

        except Exception as e:
            return False, "", f"Docker Error: {str(e)}"
        finally:
            if container:
                try:
                    container.remove(force=True)
                except:
                    pass
