import os
import requests
import zipfile
import shutil

def download_ffmpeg():
    """Download and setup ffmpeg for Windows"""
    print("🎬 Downloading FFmpeg for Windows...")
    
    # Create ffmpeg directory
    ffmpeg_dir = "ffmpeg"
    if not os.path.exists(ffmpeg_dir):
        os.makedirs(ffmpeg_dir)
    
    # FFmpeg download URL for Windows (static build)
    ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    
    try:
        print("📥 Downloading FFmpeg zip file...")
        response = requests.get(ffmpeg_url, stream=True)
        response.raise_for_status()
        
        zip_filename = "ffmpeg.zip"
        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded {zip_filename} ({os.path.getsize(zip_filename) / 1024 / 1024:.1f} MB)")
        
        print("📦 Extracting FFmpeg...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall("temp_ffmpeg")
        
        # Find the extracted folder
        temp_dir = "temp_ffmpeg"
        extracted_folders = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f))]
        
        if extracted_folders:
            extracted_folder = os.path.join(temp_dir, extracted_folders[0])
            bin_folder = os.path.join(extracted_folder, "bin")
            
            if os.path.exists(bin_folder):
                # Copy ffmpeg.exe to the backend directory
                ffmpeg_exe = os.path.join(bin_folder, "ffmpeg.exe")
                if os.path.exists(ffmpeg_exe):
                    dest_path = os.path.join("backend", "ffmpeg.exe")
                    shutil.copy2(ffmpeg_exe, dest_path)
                    print(f"✅ FFmpeg installed to {dest_path}")
                    
                    # Also copy to current directory for easier access
                    shutil.copy2(ffmpeg_exe, "ffmpeg.exe")
                    print("✅ FFmpeg copied to current directory")
                else:
                    print("❌ ffmpeg.exe not found in extracted files")
            else:
                print("❌ bin folder not found in extracted files")
        else:
            print("❌ No folders found in extracted zip")
        
        # Clean up
        os.remove(zip_filename)
        shutil.rmtree(temp_dir)
        print("🧹 Cleaned up temporary files")
        
        # Test ffmpeg
        if os.path.exists("ffmpeg.exe"):
            import subprocess
            try:
                result = subprocess.run(["ffmpeg.exe", "-version"], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ FFmpeg is working correctly!")
                    return True
                else:
                    print("❌ FFmpeg test failed")
            except Exception as e:
                print(f"❌ Error testing FFmpeg: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Error downloading FFmpeg: {e}")
        return False

if __name__ == "__main__":
    print("🛠️ Setting up FFmpeg for WebM audio conversion")
    print("=" * 50)
    
    # Check if ffmpeg already exists
    if os.path.exists("ffmpeg.exe") or os.path.exists("backend/ffmpeg.exe"):
        print("✅ FFmpeg already exists!")
    else:
        success = download_ffmpeg()
        if success:
            print("\n🎉 FFmpeg setup complete!")
            print("WebM audio conversion should now work.")
        else:
            print("\n❌ FFmpeg setup failed.")
            print("Manual installation may be required.")
    
    print("\n" + "=" * 50)
    print("Setup complete! Restart the backend server to use FFmpeg.")