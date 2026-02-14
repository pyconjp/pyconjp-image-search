import type { SearchResult } from "../types";
import { GalleryItem } from "./GalleryItem";

interface Props {
  results: SearchResult[];
  onSelect: (index: number) => void;
}

export function Gallery({ results, onSelect }: Props) {
  if (results.length === 0) return null;

  return (
    <div className="gallery">
      {results.map((result, index) => (
        <GalleryItem
          key={`${result.id}-${index}`}
          result={result}
          onClick={() => onSelect(index)}
        />
      ))}
    </div>
  );
}
