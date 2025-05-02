import React, { useState, useEffect, useRef } from 'react';

function IPhoneMessageApp() {
  const [responseMessage, setResponseMessage] = useState("Get Ready...");
  const [originalMessage, setOriginalMessage] = useState("How are you feeling today?");
  const [emotion, setEmotion] = useState("neutral");
  const [emotionProbs, setEmotionProbs] = useState({});
  const [photoTaken, setPhotoTaken] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [webcamReady, setWebcamReady] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null); // Store the stream reference

  // Initialize webcam and fetch random message from backend
  useEffect(() => {
    // Start webcam
    const startWebcam = async () => {
      try {
        console.log("Requesting webcam access...");
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 320 },
            height: { ideal: 240 },
            facingMode: "user" // Explicitly request front camera
          }
        });

        streamRef.current = stream; // Store stream reference

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          console.log("Webcam stream assigned to video element");

          videoRef.current.onloadedmetadata = () => {
            console.log("Video metadata loaded");
            videoRef.current.play()
              .then(() => {
                console.log("Video playback started");
                setWebcamReady(true);
              })
              .catch(err => {
                console.error("Error starting video playback:", err);
                setErrorMessage("Error starting video: " + err.message);
              });
          };
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
        setResponseMessage("Could not access webcam. Please check permissions.");
        setErrorMessage("Webcam error: " + err.message);
      }
    };

    startWebcam();

    // Fetch message from backend
    console.log("Fetching message from backend...");
    fetch("/get_message")
      .then(response => {
        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log("Received message from backend:", data);
        setOriginalMessage(data.original_message || data.message || "How are you feeling today?");
        setResponseMessage("Please react naturally to the message above. Taking your photo in a moment...");
      })
      .catch(error => {
        console.error("Error fetching message:", error);
        // Fallback message if fetch fails
        setOriginalMessage("How are you feeling today?");
        setResponseMessage("Say cheese! Taking your photo in a moment...");
        setErrorMessage("Message fetch error: " + error.message);
      });

    return () => {
      // Clean up video streams when component unmounts
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []); // No dependencies needed here

  // Add a separate effect to take photo after webcam is ready
  useEffect(() => {
    if (webcamReady) {
      console.log("Webcam is ready, scheduling automatic photo capture");
      const timer = setTimeout(() => {
        console.log("Taking snapshot automatically");
        takeSnapshot();
      }, 5000); // 5 seconds to give user time to prepare

      return () => clearTimeout(timer);
    }
  }, [webcamReady]);

  const takeSnapshot = () => {
    console.log("takeSnapshot called, checking prerequisites...");
    setErrorMessage("");

    if (!videoRef.current) {
      const msg = "Video reference is not available";
      console.error(msg);
      setErrorMessage(msg);
      setResponseMessage("Error: Video not available");
      return;
    }

    if (!canvasRef.current) {
      const msg = "Canvas reference is not available";
      console.error(msg);
      setErrorMessage(msg);
      setResponseMessage("Error: Canvas not available");
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Check if video is playing and has dimensions
    if (video.readyState !== 4 || video.videoWidth === 0 || video.videoHeight === 0) {
      const msg = `Video is not ready yet. readyState: ${video.readyState}, dimensions: ${video.videoWidth}x${video.videoHeight}`;
      console.error(msg);
      setErrorMessage(msg);

      // Try again after a short delay
      setTimeout(() => {
        console.log("Retrying snapshot after delay");
        takeSnapshot();
      }, 1500);
      return;
    }

    console.log("All prerequisites check out, taking snapshot now");
    setIsProcessing(true);

    try {
      const context = canvas.getContext('2d');

      // Make sure canvas dimensions match video dimensions
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      console.log(`Canvas size set to ${canvas.width}x${canvas.height}`);

      // Clear canvas first
      context.clearRect(0, 0, canvas.width, canvas.height);

      // Draw the video frame to the canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      console.log("Drew video frame to canvas");

      // Convert canvas to data URL
      const imageData = canvas.toDataURL('image/jpeg', 0.9);
      console.log("Converted canvas to JPEG data URL, length:", imageData.length);

      if (imageData.length < 100) {
        const msg = "Image data is too small, likely empty";
        console.error(msg);
        setErrorMessage(msg);
        setResponseMessage("Error capturing image. Please try again.");
        setIsProcessing(false);
        return;
      }

      setPhotoTaken(true);
      setResponseMessage("Analyzing your reaction...");

      // Send the captured image to backend
      console.log("Sending image to backend...");
      fetch('/capture_photo', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          image: imageData,
          original_message: originalMessage
        })
      })
      .then(response => {
        console.log("Received initial response from capture_photo:", response.status);
        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        return response.json();
      })
      .then(data => {
        console.log("Photo capture complete response:", data);
        if (data.success) {
          console.log("Successfully uploaded image, getting emotion response");
          // Now get the response based on the image
          return fetch('/get_response', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              image_path: data.filename,
              message: originalMessage
            })
          });
        } else {
          throw new Error(data.error || "Failed to upload image");
        }
      })
      .then(response => {
        console.log("Received response from get_response endpoint:", response.status);
        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        return response.json();
      })
      .then(responseData => {
        console.log("Emotion analysis complete, response data:", responseData);
        // Update the UI with the response
        setResponseMessage(responseData.message || "I see how you feel about that!");
        setEmotion(responseData.emotion || "neutral");
        setEmotionProbs(responseData.emotion_class_probs || {});
        setIsProcessing(false);
      })
      .catch(error => {
        console.error("Error processing photo:", error);
        setResponseMessage("Sorry, there was an error analyzing your reaction: " + error.message);
        setErrorMessage("API error: " + error.message);
        setIsProcessing(false);
      });
    } catch (err) {
      console.error("Error taking snapshot:", err);
      setResponseMessage("Error taking photo: " + err.message);
      setErrorMessage("Snapshot error: " + err.message);
      setIsProcessing(false);
    }
  };

  // Format probability as percentage
  const formatProb = (prob) => {
    return `${Math.round(prob * 100)}%`;
  };

  // Handler for manual retake photo
  const handleRetakePhoto = () => {
    setPhotoTaken(false);
    setIsProcessing(false);
    setErrorMessage("");
    setResponseMessage("Getting ready to take another photo...");

    // Take a new photo after a short delay
    setTimeout(() => {
      takeSnapshot();
    }, 2000);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="flex flex-col items-center">
        <div className="flex flex-col md:flex-row items-center justify-center gap-8 mb-4">
          {/* iPhone frame */}
          <div className="w-80 h-[600px] bg-black rounded-[40px] p-3 shadow-xl mb-4 md:mb-0">
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
                {/* Received message bubble (from random message generator) */}
                <div className="px-4 py-2">
                  <div className="bg-gray-300 text-black p-3 rounded-t-xl rounded-bl-xl max-w-[80%] mr-auto relative mb-1">
                    <p className="text-sm">{originalMessage}</p>
                  </div>
                  <span className="text-xs text-gray-500 flex justify-start pl-2">
                    Just now
                  </span>
                </div>

                {/* Response message bubble from model */}
                <div className="px-4 py-2">
                  <div className="bg-blue-500 text-white p-3 rounded-t-xl rounded-br-xl max-w-[80%] ml-auto relative mb-1">
                    <p className="text-sm">{responseMessage}</p>
                  </div>
                  <span className="text-xs text-gray-500 flex justify-end pr-2">
                    Just now
                  </span>
                </div>

                {/* Photo and emotion display */}
                {photoTaken && (
                  <div className="absolute bottom-4 left-0 right-0 px-4">
                    <div className="bg-white rounded-xl shadow-md overflow-hidden">
                      <canvas
                        ref={canvasRef}
                        width={320}
                        height={240}
                        className="w-full h-32 object-cover"
                      />
                      {emotion && (
                        <div className="bg-gray-100 p-2 text-center">
                          <p className="text-sm font-medium">Emotion: <span className="font-bold text-blue-500">{emotion}</span></p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Camera preview and emotion probabilities */}
          <div className="flex flex-col gap-2">
            {/* Camera preview */}
            <div className="bg-black p-2 rounded-xl shadow-md">
              <div className="text-white text-center text-sm mb-2">
                Camera Preview {webcamReady ? "(Ready)" : "(Initializing...)"}
              </div>
              <video
                ref={videoRef}
                width={320}
                height={240}
                autoPlay
                playsInline
                muted
                className="rounded-lg"
              />
            </div>

            {/* Emotion probabilities */}
            {photoTaken && Object.keys(emotionProbs).length > 0 && (
              <div className="bg-white p-3 rounded-xl shadow-md">
                <h3 className="text-sm font-bold mb-2">Emotion Probabilities:</h3>
                <div className="grid grid-cols-1 gap-2">
                  {Object.entries(emotionProbs).sort((a, b) => b[1] - a[1]).map(([emotion, prob]) => (
                    <div key={emotion} className="flex items-center">
                      <span className="text-xs font-medium w-24 capitalize">{emotion}:</span>
                      <div className="flex-1 h-4 bg-gray-200 rounded overflow-hidden">
                        <div
                          className="h-full bg-blue-500"
                          style={{ width: `${Math.round(prob * 100)}%` }}
                        ></div>
                      </div>
                      <span className="text-xs ml-2 w-12 text-right">{formatProb(prob)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {isProcessing && (
              <div className="bg-white p-4 rounded-xl shadow-md text-center">
                <p className="text-sm">Processing...</p>
              </div>
            )}

            {errorMessage && (
              <div className="bg-red-100 border border-red-400 text-red-700 p-2 rounded text-xs">
                {errorMessage}
              </div>
            )}
          </div>
        </div>

        {/* Control buttons */}
        <div className="flex space-x-4">
          <button
            onClick={handleRetakePhoto}
            className="mt-4 px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition">
            Retake Photo
          </button>

          <button
            onClick={() => {
              console.log("Manual capture button clicked");
              takeSnapshot();
            }}
            className="mt-4 px-6 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition">
            Capture Now
          </button>
        </div>

        {/* Debug information */}
        <div className="mt-4 p-2 bg-gray-200 rounded text-xs text-gray-800 max-w-xl">
          <p>Webcam status: {webcamReady ? "Ready" : "Initializing"}</p>
          <p>Video element: {videoRef.current ? `${videoRef.current.videoWidth}x${videoRef.current.videoHeight}` : "Not available"}</p>
          <p>Canvas element: {canvasRef.current ? `${canvasRef.current.width}x${canvasRef.current.height}` : "Not available"}</p>
        </div>
      </div>
    </div>
  );
}

export default IPhoneMessageApp;