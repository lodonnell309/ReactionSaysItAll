import React, { useState, useEffect, useRef } from 'react';

function IPhoneMessageApp() {
  const [message, setMessage] = useState("Get Ready...");
  const [photoTaken, setPhotoTaken] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Initialize webcam and fetch random message from backend
  useEffect(() => {
    // Start webcam
    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    };

    startWebcam();

    // Fetch random message from backend
    fetch("/get_message")
      .then(response => response.json())
      .then(data => {
        setMessage(data.message);
        // Set timeout to take photo after message is displayed
        setTimeout(() => {
          takeSnapshot();
        }, 4000); // 4 seconds delay
      })
      .catch(error => {
        console.error("Error fetching message:", error);
        // Fallback message if fetch fails
        setMessage("Say cheese! Taking your photo in 4 seconds...");
        setTimeout(() => {
          takeSnapshot();
        }, 4000);
      });

    return () => {
      // Clean up video streams when component unmounts
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const takeSnapshot = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      // Draw the video frame to the canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert canvas to data URL
      const imageData = canvas.toDataURL('image/jpeg');

      setPhotoTaken(true);

      // Send the captured image to backend
      fetch('/capture_photo', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData })
      })
      .then(response => {
        console.log("Photo sent to backend");
        // No longer changing the message here
      })
      .catch(error => {
        console.error("Error sending photo:", error);
        // No longer changing the message here either
      });
    }
  };

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      <div className="flex items-center justify-center gap-8">
        {/* iPhone frame */}
        <div className="w-80 h-[600px] bg-black rounded-[40px] p-3 shadow-xl">
          {/* iPhone screen */}
          <div className="bg-white h-full w-full rounded-[32px] overflow-hidden relative">
            {/* Status bar */}
            <div className="h-7 bg-gray-100 flex justify-between items-center px-5">
              <span className="text-xs font-semibold">9:41</span>
              <div className="flex items-center gap-1">
                <div className="w-4 h-2.5 bg-black rounded-sm"></div>
                <div className="w-3 h-3 bg-black rounded-full"></div>
              </div>
            </div>

            {/* Message app background */}
            <div className="h-full w-full pt-4 bg-gray-100 flex flex-col">
              {/* Message bubble from backend */}
              <div className="px-4 py-2">
                <div className="bg-blue-500 text-white p-3 rounded-t-xl rounded-br-xl max-w-[80%] ml-auto relative mb-1">
                  <p className="text-sm">{message}</p>
                </div>
                <span className="text-xs text-gray-500 flex justify-end pr-2">
                  Just now
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Camera preview (separate from iPhone) */}
        <div className="bg-black p-2 rounded-xl shadow-md">
          <div className="text-white text-center text-sm mb-2">Camera Preview</div>
          <video
            ref={videoRef}
            width={320}
            height={240}
            autoPlay
            muted
            className="rounded-lg"
          />
          <canvas
            ref={canvasRef}
            width={320}
            height={240}
            className="hidden"
          />
        </div>
      </div>
    </div>
  );
}

export default IPhoneMessageApp;