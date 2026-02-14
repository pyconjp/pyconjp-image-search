import { useEffect, useRef, useState } from "react";
import { CLIPEncoder } from "../lib/clip";

export function useCLIPEncoder() {
  const encoderRef = useRef<CLIPEncoder | null>(null);
  const [isTextReady, setIsTextReady] = useState(false);
  const [isVisionReady, setIsVisionReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const encoder = new CLIPEncoder();
    encoderRef.current = encoder;

    encoder
      .loadTextModel((p) => {
        if (!cancelled) setProgress(p);
      })
      .then(() => {
        if (!cancelled) {
          setIsTextReady(true);
          setIsLoading(false);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const loadVisionModel = async () => {
    if (!encoderRef.current || isVisionReady) return;
    await encoderRef.current.loadVisionModel();
    setIsVisionReady(true);
  };

  return {
    encoder: encoderRef.current,
    isTextReady,
    isVisionReady,
    isLoading,
    progress,
    error,
    loadVisionModel,
  };
}
