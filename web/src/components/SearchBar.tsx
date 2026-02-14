import { useState, useRef, useEffect, type FormEvent } from "react";

interface Props {
  onSearch: (query: string) => void;
  isSearching: boolean;
  disabled: boolean;
}

const hasTranslatorAPI = typeof Translator !== "undefined";

export function SearchBar({ onSearch, isSearching, disabled }: Props) {
  const [query, setQuery] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);
  const [translatorAvailable, setTranslatorAvailable] = useState(false);
  const translatorRef = useRef<TranslatorInstance | null>(null);

  // Check if ja→en translation is available
  useEffect(() => {
    if (!hasTranslatorAPI) return;
    Translator!
      .availability({ sourceLanguage: "ja", targetLanguage: "en" })
      .then((status) => {
        setTranslatorAvailable(
          status === "available" || status === "downloadable",
        );
      })
      .catch(() => setTranslatorAvailable(false));
  }, []);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  const handleTranslate = async () => {
    if (!query.trim() || !hasTranslatorAPI) return;
    setIsTranslating(true);
    try {
      if (!translatorRef.current) {
        translatorRef.current = await Translator!.create({
          sourceLanguage: "ja",
          targetLanguage: "en",
        });
      }
      const translated = await translatorRef.current.translate(query.trim());
      setQuery(translated);
    } catch (err) {
      console.error("Translation failed:", err);
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="e.g. keynote speaker on stage"
        disabled={disabled}
        className="search-input"
      />
      <button
        type="button"
        onClick={handleTranslate}
        disabled={
          !translatorAvailable ||
          disabled ||
          isSearching ||
          isTranslating ||
          !query.trim()
        }
        className="translate-button"
        title={
          translatorAvailable
            ? "日本語を英語に翻訳"
            : "Chrome 138+ で利用可能です"
        }
      >
        {isTranslating ? "翻訳中..." : "英訳"}
      </button>
      <button
        type="submit"
        disabled={disabled || isSearching || !query.trim()}
        className="search-button"
      >
        {isSearching ? "Searching..." : "Search"}
      </button>
    </form>
  );
}
