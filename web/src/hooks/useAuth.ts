import {
  signOut as firebaseSignOut,
  GoogleAuthProvider,
  onAuthStateChanged,
  signInWithPopup,
  type User,
} from "firebase/auth";
import { useEffect, useState } from "react";
import { auth } from "../lib/firebase";

const AUTH_REQUIRED = import.meta.env.VITE_AUTH_REQUIRED === "true";

interface AuthState {
  user: User | null;
  loading: boolean;
  error: string | null;
  signIn: () => Promise<void>;
  signOut: () => Promise<void>;
}

export function useAuth(): AuthState {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(AUTH_REQUIRED);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!AUTH_REQUIRED) {
      setLoading(false);
      return;
    }

    const unsubscribe = onAuthStateChanged(auth, (u) => {
      setUser(u);
      setLoading(false);
    });
    return unsubscribe;
  }, []);

  const signIn = async () => {
    const provider = new GoogleAuthProvider();
    setError(null);
    try {
      const result = await signInWithPopup(auth, provider);
      const email = result.user.email ?? "";
      if (!email.endsWith("@pycon.jp")) {
        await firebaseSignOut(auth);
        setError("@pycon.jp のアカウントでログインしてください。");
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error("signIn error:", msg);
      setError(msg);
    }
  };

  const signOut = async () => {
    await firebaseSignOut(auth);
  };

  return { user, loading, error, signIn, signOut };
}

export { AUTH_REQUIRED };
