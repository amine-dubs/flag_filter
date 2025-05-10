import cv2
import numpy as np
import os
import time
import random
import datetime
import threading
import wave
import pyaudio
import subprocess
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import shutil
from pathlib import Path

class TikTokFlagFilter:
    def __init__(self):
        self.cap = None
        self.face_cascade = None
        self.flags_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flags")
        self.flags = []
        self.current_flag = None
        self.flag_start_time = 0
        self.show_flag_name = False
        self.game_active = False
        # Recording related variables
        self.recording = False
        self.video_writer = None
        self.output_file = None
        self.audio_output = None
        self.audio_thread = None
        self.audio_stream = None
        self.pyaudio_instance = None
        self.audio_frames = []
        self.timestamp = None
        
    def setup(self):
        """Set up video capture and load resources"""
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
            
        # Load face detection cascade
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(face_cascade_path):
            print(f"Error: Face cascade file not found at {face_cascade_path}")
            return False
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Ensure flags directory exists
        if not os.path.exists(self.flags_dir):
            os.makedirs(self.flags_dir)
            self.download_sample_flags()
        
        # Load flags
        self.load_flags()
        if not self.flags:
            print("Error: No flag images found")
            return False
            
        return True
    
    def download_sample_flags(self):
        """Download some sample flags if none exist"""
        sample_flags = {
            'usa.png': 'United States',
            'uk.png': 'United Kingdom',
            'france.png': 'France',
            'germany.png': 'Germany',
            'japan.png': 'Japan',
            'brazil.png': 'Brazil',
            'canada.png': 'Canada',
            'australia.png': 'Australia',
            'india.png': 'India',
            'mexico.png': 'Mexico'
        }
        
        # Base URL for flag images (using a free flag API)
        base_url = "https://flagcdn.com/w320/"
        
        country_codes = {
            'United States': 'us',
            'United Kingdom': 'gb',
            'France': 'fr',
            'Germany': 'de',
            'Japan': 'jp',
            'Brazil': 'br',
            'Canada': 'ca',
            'Australia': 'au',
            'India': 'in',
            'Mexico': 'mx'
        }
        
        print("Downloading sample flags...")
        for filename, flag_name in sample_flags.items():
            try:
                code = country_codes[flag_name].lower()
                url = f"{base_url}{code}.png"
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    filepath = os.path.join(self.flags_dir, filename)
                    with open(filepath, 'wb') as f:
                        response.raw.decode_content = True
                        shutil.copyfileobj(response.raw, f)
                    
                    # Create a metadata file with the flag name
                    meta_path = os.path.splitext(filepath)[0] + ".txt"
                    with open(meta_path, 'w') as f:
                        f.write(flag_name)
                        
                    print(f"Downloaded {filename}")
                else:
                    print(f"Failed to download {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
    
    def download_more_flags(self):
        """Download additional flags to expand the collection"""
        more_flags = {
            'italy.png': 'Italy',
            'spain.png': 'Spain',
            'china.png': 'China',
            'russia.png': 'Russia',
            'south_korea.png': 'South Korea',
            'south_africa.png': 'South Africa',
            'argentina.png': 'Argentina',
            'sweden.png': 'Sweden',
            'norway.png': 'Norway',
            'finland.png': 'Finland',
            'netherlands.png': 'Netherlands',
            'belgium.png': 'Belgium',
            'switzerland.png': 'Switzerland',
            'portugal.png': 'Portugal',
            'egypt.png': 'Egypt',
            'nigeria.png': 'Nigeria',
            'kenya.png': 'Kenya',
            'thailand.png': 'Thailand',
            'singapore.png': 'Singapore',
            'new_zealand.png': 'New Zealand',
            'greece.png': 'Greece',
            'turkey.png': 'Turkey',
            'ireland.png': 'Ireland',
            'vietnam.png': 'Vietnam',
            'philippines.png': 'Philippines',
            # Adding 25 more countries
            'saudi_arabia.png': 'Saudi Arabia',
            'indonesia.png': 'Indonesia',
            'pakistan.png': 'Pakistan',
            'ukraine.png': 'Ukraine',
            'colombia.png': 'Colombia',
            'malaysia.png': 'Malaysia',
            'chile.png': 'Chile',
            'denmark.png': 'Denmark',
            'israel.png': 'Israel',
            'austria.png': 'Austria',
            'hungary.png': 'Hungary',
            'czech_republic.png': 'Czech Republic',
            'peru.png': 'Peru',
            'romania.png': 'Romania',
            'morocco.png': 'Morocco',
            'bangladesh.png': 'Bangladesh',
            'taiwan.png': 'Taiwan',
            'poland.png': 'Poland',
            'croatia.png': 'Croatia',
            'serbia.png': 'Serbia',
            'iceland.png': 'Iceland',
            'jordan.png': 'Jordan',
            'lebanon.png': 'Lebanon',
            'uae.png': 'United Arab Emirates',
            'kuwait.png': 'Kuwait'
        }
        
        # Base URL for flag images (using a free flag API)
        base_url = "https://flagcdn.com/w320/"
        
        country_codes = {
            'Italy': 'it',
            'Spain': 'es',
            'China': 'cn',
            'Russia': 'ru',
            'South Korea': 'kr',
            'South Africa': 'za',
            'Argentina': 'ar',
            'Sweden': 'se',
            'Norway': 'no',
            'Finland': 'fi',
            'Netherlands': 'nl',
            'Belgium': 'be',
            'Switzerland': 'ch',
            'Portugal': 'pt',
            'Egypt': 'eg',
            'Nigeria': 'ng',
            'Kenya': 'ke',
            'Thailand': 'th',
            'Singapore': 'sg',
            'New Zealand': 'nz',
            'Greece': 'gr',
            'Turkey': 'tr',
            'Ireland': 'ie',
            'Vietnam': 'vn',
            'Philippines': 'ph',
            # Adding corresponding country codes for the new countries
            'Saudi Arabia': 'sa',
            'Indonesia': 'id',
            'Pakistan': 'pk',
            'Ukraine': 'ua',
            'Colombia': 'co',
            'Malaysia': 'my',
            'Chile': 'cl',
            'Denmark': 'dk',
            'Israel': 'il',
            'Austria': 'at',
            'Hungary': 'hu',
            'Czech Republic': 'cz',
            'Peru': 'pe',
            'Romania': 'ro',
            'Morocco': 'ma',
            'Bangladesh': 'bd',
            'Taiwan': 'tw',
            'Poland': 'pl',
            'Croatia': 'hr',
            'Serbia': 'rs',
            'Iceland': 'is',
            'Jordan': 'jo',
            'Lebanon': 'lb',
            'United Arab Emirates': 'ae',
            'Kuwait': 'kw'
        }
        
        print("Downloading additional flags...")
        downloaded_count = 0
        for filename, flag_name in more_flags.items():
            # Check if file already exists
            filepath = os.path.join(self.flags_dir, filename)
            if os.path.exists(filepath):
                print(f"Flag {filename} already exists, skipping.")
                continue
                
            try:
                code = country_codes[flag_name].lower()
                url = f"{base_url}{code}.png"
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        response.raw.decode_content = True
                        shutil.copyfileobj(response.raw, f)
                    
                    # Create a metadata file with the flag name
                    meta_path = os.path.splitext(filepath)[0] + ".txt"
                    with open(meta_path, 'w') as f:
                        f.write(flag_name)
                        
                    downloaded_count += 1
                    print(f"Downloaded {filename}")
                else:
                    print(f"Failed to download {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                
        print(f"Downloaded {downloaded_count} new flags.")
        if downloaded_count > 0:
            # Reload flags to include the new ones
            self.load_flags()
            return True
        return False
    
    def download_challenging_flags(self):
        """Download challenging flags from lesser-known countries and territories"""
        challenging_flags = {
            'monaco.png': 'Monaco',
            'vatican.png': 'Vatican City',
            'liechtenstein.png': 'Liechtenstein',
            'andorra.png': 'Andorra',
            'san_marino.png': 'San Marino',
            'malta.png': 'Malta',
            'bahrain.png': 'Bahrain',
            'qatar.png': 'Qatar',
            'bhutan.png': 'Bhutan',
            'brunei.png': 'Brunei',
            'fiji.png': 'Fiji',
            'tonga.png': 'Tonga',
            'samoa.png': 'Samoa',
            'vanuatu.png': 'Vanuatu',
            'palau.png': 'Palau',
            'nauru.png': 'Nauru',
            'kiribati.png': 'Kiribati',
            'tuvalu.png': 'Tuvalu',
            'maldives.png': 'Maldives',
            'suriname.png': 'Suriname',
            'guyana.png': 'Guyana',
            'belize.png': 'Belize',
            'barbados.png': 'Barbados',
            'bahamas.png': 'Bahamas',
            'antigua_barbuda.png': 'Antigua and Barbuda',
            'grenada.png': 'Grenada',
            'st_lucia.png': 'Saint Lucia',
            'dominica.png': 'Dominica',
            'st_vincent.png': 'Saint Vincent and the Grenadines',
            'montenegro.png': 'Montenegro',
            'slovenia.png': 'Slovenia',
            'macedonia.png': 'North Macedonia',
            'albania.png': 'Albania',
            'moldova.png': 'Moldova',
            'estonia.png': 'Estonia',
            'latvia.png': 'Latvia',
            'lithuania.png': 'Lithuania',
            'belarus.png': 'Belarus',
            'armenia.png': 'Armenia',
            'georgia.png': 'Georgia'
        }
        
        # Base URL for flag images (using a free flag API)
        base_url = "https://flagcdn.com/w320/"
        
        country_codes = {
            'Monaco': 'mc',
            'Vatican City': 'va',
            'Liechtenstein': 'li',
            'Andorra': 'ad',
            'San Marino': 'sm',
            'Malta': 'mt',
            'Bahrain': 'bh',
            'Qatar': 'qa',
            'Bhutan': 'bt',
            'Brunei': 'bn',
            'Fiji': 'fj',
            'Tonga': 'to',
            'Samoa': 'ws',
            'Vanuatu': 'vu',
            'Palau': 'pw',
            'Nauru': 'nr',
            'Kiribati': 'ki',
            'Tuvalu': 'tv',
            'Maldives': 'mv',
            'Suriname': 'sr',
            'Guyana': 'gy',
            'Belize': 'bz',
            'Barbados': 'bb',
            'Bahamas': 'bs',
            'Antigua and Barbuda': 'ag',
            'Grenada': 'gd',
            'Saint Lucia': 'lc',
            'Dominica': 'dm',
            'Saint Vincent and the Grenadines': 'vc',
            'Montenegro': 'me',
            'Slovenia': 'si',
            'North Macedonia': 'mk',
            'Albania': 'al',
            'Moldova': 'md',
            'Estonia': 'ee',
            'Latvia': 'lv',
            'Lithuania': 'lt',
            'Belarus': 'by',
            'Armenia': 'am',
            'Georgia': 'ge'
        }
        
        print("Downloading challenging flags...")
        downloaded_count = 0
        for filename, flag_name in challenging_flags.items():
            # Check if file already exists
            filepath = os.path.join(self.flags_dir, filename)
            if os.path.exists(filepath):
                print(f"Flag {filename} already exists, skipping.")
                continue
                
            try:
                code = country_codes[flag_name].lower()
                url = f"{base_url}{code}.png"
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        response.raw.decode_content = True
                        shutil.copyfileobj(response.raw, f)
                    
                    # Create a metadata file with the flag name
                    meta_path = os.path.splitext(filepath)[0] + ".txt"
                    with open(meta_path, 'w') as f:
                        f.write(flag_name)
                        
                    downloaded_count += 1
                    print(f"Downloaded {filename}")
                else:
                    print(f"Failed to download {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                
        print(f"Downloaded {downloaded_count} new challenging flags.")
        if downloaded_count > 0:
            # Reload flags to include the new ones
            self.load_flags()
            return True
        return False
    
    def load_flags(self):
        """Load flag images and names from the flags directory"""
        self.flags = []
        for file in os.listdir(self.flags_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                flag_path = os.path.join(self.flags_dir, file)
                meta_path = os.path.splitext(flag_path)[0] + ".txt"
                
                # Get flag name from metadata file or filename
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        flag_name = f.read().strip()
                else:
                    flag_name = os.path.splitext(file)[0].replace('_', ' ').title()
                
                # Try to load the image
                img = cv2.imread(flag_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.flags.append({
                        'path': flag_path,
                        'name': flag_name,
                        'image': img
                    })
                else:
                    print(f"Warning: Failed to load image {flag_path}")
        
        print(f"Loaded {len(self.flags)} flags.")
        
    def select_random_flag(self):
        """Select a random flag"""
        self.current_flag = random.choice(self.flags)
        self.flag_start_time = time.time()
        self.show_flag_name = False
        self.game_active = True
        
    def resize_flag(self, flag_img, width):
        """Resize flag while maintaining aspect ratio"""
        height = int(flag_img.shape[0] * (width / flag_img.shape[1]))
        return cv2.resize(flag_img, (width, height))
        
    def overlay_flag(self, frame, face):
        """Overlay the flag above the detected face"""
        if self.current_flag is None:
            return frame
            
        # Extract face coordinates
        x, y, w, h = face
        
        # Resize flag to match face width
        flag_img = self.resize_flag(self.current_flag['image'], w)
        flag_h, flag_w = flag_img.shape[:2]
        
        # Calculate position (above the head)
        flag_x = x
        flag_y = max(0, y - flag_h - 20)  # 20 pixels above the face
        
        # Overlay flag on frame
        if flag_img.shape[2] == 4:  # With alpha channel
            # Create a region of interest
            roi = frame[flag_y:flag_y+flag_h, flag_x:flag_x+flag_w]
            
            # Create a mask from the alpha channel
            alpha_channel = flag_img[:, :, 3] / 255.0
            alpha_3_channel = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=2)
            
            # Apply the mask
            foreground = flag_img[:, :, :3] * alpha_3_channel
            background = roi * (1 - alpha_3_channel)
            result = foreground + background
            
            # Put result back into the frame
            frame[flag_y:flag_y+flag_h, flag_x:flag_x+flag_w] = result.astype(np.uint8)
        else:
            # Simple overlay without transparency
            flag_area = frame[flag_y:flag_y+flag_h, flag_x:flag_x+flag_w]
            if flag_area.shape[0] == flag_img.shape[0] and flag_area.shape[1] == flag_img.shape[1]:
                frame[flag_y:flag_y+flag_h, flag_x:flag_x+flag_w] = flag_img
        
        # Check if it's time to show the flag name
        elapsed_time = time.time() - self.flag_start_time
        if elapsed_time >= 5 and not self.show_flag_name:
            self.show_flag_name = True
            
        # Show flag name if time has elapsed
        if self.show_flag_name:
            cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)
            draw = ImageDraw.Draw(pil_im)
            
            # Use a default font
            try:
                font = ImageFont.truetype("arial.ttf", 36)
            except IOError:
                font = ImageFont.load_default()
                
            # Draw text with a background
            text = self.current_flag['name']
            
            # Fix: Replace deprecated textsize with getbbox or getsize
            try:
                # For newer Pillow versions
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                try:
                    # For slightly older Pillow versions
                    text_width, text_height = font.getsize(text)
                except AttributeError:
                    # Fallback for very old versions (unlikely)
                    text_width, text_height = 200, 40  # Default estimated size
            
            text_x = (frame.shape[1] - text_width) // 2
            text_y = frame.shape[0] - text_height - 30
            
            # Draw background rectangle
            draw.rectangle(
                [(text_x-10, text_y-10), (text_x+text_width+10, text_y+text_height+10)],
                fill=(0, 0, 0, 200)
            )
            
            # Draw text
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
            
            # Convert back to OpenCV format
            frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            
        return frame
        
    def start_audio_recording(self):
        """Start recording audio in a separate thread"""
        self.pyaudio_instance = pyaudio.PyAudio()
        
        # Set audio parameters
        self.audio_frames = []
        audio_format = pyaudio.paInt16
        channels = 1
        rate = 44100
        chunk = 1024
        
        def audio_callback(in_data, frame_count, time_info, status):
            if self.recording:
                self.audio_frames.append(in_data)
            return (in_data, pyaudio.paContinue)
        
        # Open audio stream
        self.audio_stream = self.pyaudio_instance.open(
            format=audio_format,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
            stream_callback=audio_callback
        )
        
        self.audio_stream.start_stream()
        print("Audio recording started.")
    
    def stop_audio_recording(self):
        """Stop and save audio recording"""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
            
        if self.pyaudio_instance:
            # Save audio to WAV file
            if self.audio_frames:
                audio_output = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          f"flag_challenge_audio_{self.timestamp}.wav")
                
                wf = wave.open(audio_output, 'wb')
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()
                
                self.audio_output = audio_output
                print(f"Audio saved to {self.audio_output}")
            
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
    
    def merge_audio_video(self):
        """Merge the audio and video files using FFmpeg"""
        if not self.output_file or not self.audio_output:
            print("Missing audio or video file for merging")
            return False
            
        try:
            # Final output file with audio
            final_output = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      f"flag_challenge_with_audio_{self.timestamp}.mp4")
            
            # Use FFmpeg to merge video and audio
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', self.output_file,  # Video input
                '-i', self.audio_output,  # Audio input
                '-c:v', 'copy',          # Copy video stream
                '-c:a', 'aac',           # Convert audio to AAC
                '-strict', 'experimental',
                '-y',                     # Overwrite output file if it exists
                final_output
            ]
            
            subprocess.run(ffmpeg_cmd, check=True)
            
            print(f"Video with audio saved to {final_output}")
            
            # Clean up temporary files (optional)
            # os.remove(self.output_file)
            # os.remove(self.audio_output)
            
            return True
        except Exception as e:
            print(f"Error merging audio and video: {e}")
            return False
    
    def start_recording(self):
        """Start recording video and audio"""
        if self.recording:
            return False
        
        # Create a unique timestamp for this recording
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up video recording
        self.output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      f"flag_challenge_{self.timestamp}.mp4")
        
        # Get frame properties
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:  # Handle case where fps is not properly detected
            fps = 30
        
        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        self.video_writer = cv2.VideoWriter(
            self.output_file, fourcc, fps, (frame_width, frame_height)
        )
        
        if not self.video_writer.isOpened():
            print("Failed to initialize video writer")
            return False
            
        # Start audio recording in a separate thread
        self.start_audio_recording()
        
        self.recording = True
        print("Recording started with audio and video...")
        return True
    
    def stop_recording(self):
        """Stop recording and save video with audio"""
        if not self.recording:
            return
        
        self.recording = False
        
        # Stop and save video
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print(f"Video saved to {self.output_file}")
        
        # Stop and save audio
        self.stop_audio_recording()
        
        # Try to merge audio and video if both were saved
        if self.output_file and self.audio_output:
            self.merge_audio_video()
    
    def run(self):
        """Run the TikTok flag filter"""
        if not self.setup():
            return
            
        print("TikTok Flag Filter Challenge")
        print("============================")
        print("Press 'SPACE' to start a new flag challenge")
        print("Press 'A' to add more flags")
        print("Press 'C' to add challenging flags")
        print("Press 'R' to start/stop recording (with audio)")
        print("Press 'Q' to quit (saves recording if active)")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            # Flip the frame horizontally for a selfie-view
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # Draw rectangle around faces and overlay flag
            for face in faces:
                if self.game_active:
                    frame = self.overlay_flag(frame, face)
                
                # Draw rectangle around face (optional)
                x, y, w, h = face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # If no game is active and we have faces, show instruction
            if not self.game_active and len(faces) > 0:
                cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im_rgb)
                draw = ImageDraw.Draw(pil_im)
                
                text = "Press SPACE to start! | A for more flags | R to record"
                try:
                    font = ImageFont.truetype("arial.ttf", 36)
                except IOError:
                    font = ImageFont.load_default()
                
                # Fix: Replace deprecated textsize with getbbox or getsize
                try:
                    # For newer Pillow versions
                    bbox = font.getbbox(text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    try:
                        # For slightly older Pillow versions
                        text_width, text_height = font.getsize(text)
                    except AttributeError:
                        # Fallback for very old versions (unlikely)
                        text_width, text_height = 200, 40  # Default estimated size
                
                text_x = (frame.shape[1] - text_width) // 2
                text_y = 30
                
                draw.rectangle(
                    [(text_x-10, text_y-10), (text_x+text_width+10, text_y+text_height+10)],
                    fill=(0, 0, 0, 200)
                )
                draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
                
                frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            
            # Add recording indicator when active
            if self.recording:
                # Draw a red circle and "REC" text
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle
                cv2.putText(frame, "REC", (50, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Write the frame to the video file
                if self.video_writer:
                    self.video_writer.write(frame)
            
            # Display the resulting frame
            cv2.imshow('TikTok Flag Filter Challenge', frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Ensure we save the recording before quitting
                if self.recording:
                    print("Saving recording before exit...")
                    self.stop_recording()
                break
            elif key == 32:  # Space key
                self.select_random_flag()
            elif key == ord('a'):  # 'A' key to download more flags
                print("Downloading more flags...")
                if self.download_more_flags():
                    print("New flags added and loaded!")
                else:
                    print("No new flags added.")
            elif key == ord('c'):  # 'C' key to download challenging flags
                print("Downloading challenging flags...")
                if self.download_challenging_flags():
                    print("New challenging flags added and loaded!")
                else:
                    print("No new challenging flags added.")
            elif key == ord('r'):  # 'R' key to toggle recording
                if self.recording:
                    self.stop_recording()
                    print("Recording stopped.")
                else:
                    if self.start_recording():
                        print("Started recording...")
                    else:
                        print("Failed to start recording.")
        
        # Make sure recording stops if still active
        if self.recording:
            self.stop_recording()
            
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    filter_app = TikTokFlagFilter()
    filter_app.run()
