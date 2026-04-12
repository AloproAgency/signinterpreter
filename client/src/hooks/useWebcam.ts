import { useRef, useEffect, useCallback, useState } from 'react';

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [active, setActive] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setActive(true);
    } catch (err) {
      console.error('Webcam error:', err);
    }
  }, []);

  const stop = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setActive(false);
  }, []);

  const captureFrame = useCallback((): Blob | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !active) return null;

    // Square crop from center of video
    const size = 480;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // Crop center of video to square, NO mirror in canvas
    // The canvas sends raw orientation to the server
    const sx = (video.videoWidth - video.videoHeight) / 2;
    ctx.drawImage(video, sx, 0, video.videoHeight, video.videoHeight, 0, 0, size, size);

    // Convert to JPEG blob synchronously via toBlob workaround
    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
    const binary = atob(dataUrl.split(',')[1]);
    const array = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) array[i] = binary.charCodeAt(i);
    return new Blob([array], { type: 'image/jpeg' });
  }, [active]);

  useEffect(() => {
    return () => stop();
  }, [stop]);

  return { videoRef, canvasRef, active, start, stop, captureFrame };
}
