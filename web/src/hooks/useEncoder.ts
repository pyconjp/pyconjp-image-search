import { useEffect, useRef, useState } from "react";
import { VisionLanguageEncoder } from "../lib/encoder";
import type { ModelConfig } from "../lib/models";

export function useEncoder(config: ModelConfig | null) {
  const encoderRef = useRef<VisionLanguageEncoder | null>(null);
  const [isTextReady, setIsTextReady] = useState(false);
  const [isVisionReady, setIsVisionReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!config) {
      encoderRef.current = null;
      setIsTextReady(false);
      setIsVisionReady(false);
      setIsLoading(false);
      setProgress(0);
      return;
    }

    let cancelled = false;
    const encoder = new VisionLanguageEncoder(config);
    encoderRef.current = encoder;
    setIsTextReady(false);
    setIsVisionReady(false);
    setIsLoading(true);
    setProgress(0);
    setError(null);

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
  }, [config]);

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
