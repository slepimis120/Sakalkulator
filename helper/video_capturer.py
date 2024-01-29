import cv2
import keyboard


def record_video(output_path='./data/captured_video.mp4', device_index=0):
    # Open the video capture device
    cap = cv2.VideoCapture(device_index)

    # Get the default frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    recording = False
    print("Unesite R da biste započeli snimanje videa i X da biste završili")

    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imshow('Video', frame)

            # Check if 'R' key is pressed
            if keyboard.is_pressed('R') and not recording:
                print("Snimanje počelo...")
                recording = True

            # Check if 'X' key is pressed
            if keyboard.is_pressed('X') and recording:
                print("Snimanje zaustavljeno.")
                recording = False
                break

            # Write the frame if recording
            if recording:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
