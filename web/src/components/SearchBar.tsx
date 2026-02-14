import { useState, type FormEvent } from "react";

interface Props {
  onSearch: (query: string) => void;
  isSearching: boolean;
  disabled: boolean;
}

export function SearchBar({ onSearch, isSearching, disabled }: Props) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
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
        type="submit"
        disabled={disabled || isSearching || !query.trim()}
        className="search-button"
      >
        {isSearching ? "Searching..." : "Search"}
      </button>
    </form>
  );
}
