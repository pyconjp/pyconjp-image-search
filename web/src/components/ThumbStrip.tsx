import { flickrUrlResize } from "../lib/flickr";
import type { SearchResult } from "../types";

interface Props {
  results: SearchResult[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}

export function ThumbStrip({ results, selectedIndex, onSelect }: Props) {
  if (results.length === 0) return null;

  return (
    <div className="thumb-strip">
      {results.map((result, index) => (
        <img
          key={`${result.id}-${index}`}
          src={flickrUrlResize(result.image_url, "q")}
          alt={result.event_name}
          className={`thumb-strip-item ${index === selectedIndex ? "selected" : ""}`}
          onClick={() => onSelect(index)}
          loading="lazy"
        />
      ))}
    </div>
  );
}
