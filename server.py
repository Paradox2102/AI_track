from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import os

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f'Received a GET request',self.path)
    def do_POST(self):
        print(f'Received a POST request', self.path)
        try:
            # Parse the form data posted
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )

            # Get the file data from the form
            file_item = form['file']

            # Check if the file was uploaded
            if file_item.filename:
                # Specify the directory to save the uploaded file
                upload_dir = "uploads"
                os.makedirs(upload_dir, exist_ok=True)

                # Build the full path to save the file
                file_path = os.path.join(upload_dir, os.path.basename(file_item.filename))

                # Save the file to disk
                with open(file_path, 'wb') as file:
                    file.write(file_item.file.read())

                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Success: File uploaded and saved.')
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Error: No file uploaded.')

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'Error: {str(e)}'.encode('utf-8'))

if __name__ == '__main__':
    PORT = 8080
    server_address = ('', PORT)

    with HTTPServer(server_address, RequestHandler) as httpd:
        print(f'Starting server on port {PORT}...')
        httpd.serve_forever()
