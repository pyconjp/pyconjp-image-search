import { flickrUrlResize } from "../lib/flickr";
import type { SearchResult } from "../types";

interface Props {
  result: SearchResult;
  onClick: () => void;
}

export function GalleryItem({ result, onClick }: Props) {
  const thumbUrl = flickrUrlResize(result.image_url, "z");

  return (
    <div className="gallery-item" onClick={onClick}>
      <img src={thumbUrl} alt={result.event_name} loading="lazy" />
      <div className="gallery-item-caption">
        <span className="score">{result.score.toFixed(3)}</span>
        <span className="event">{result.event_name}</span>
      </div>
    </div>
  );
}
