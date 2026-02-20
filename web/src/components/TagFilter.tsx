import { useState } from "react";

interface Props {
  tags: string[];
  selectedTags: string[];
  onChange: (tags: string[]) => void;
}

const COLLAPSED_LIMIT = 15;

export function TagFilter({ tags, selectedTags, onChange }: Props) {
  const [expanded, setExpanded] = useState(false);

  const handleToggle = (tag: string) => {
    if (selectedTags.includes(tag)) {
      onChange(selectedTags.filter((t) => t !== tag));
    } else {
      onChange([...selectedTags, tag]);
    }
  };

  if (tags.length === 0) return null;

  const needsCollapse = tags.length > COLLAPSED_LIMIT;
  const visibleTags =
    needsCollapse && !expanded ? tags.slice(0, COLLAPSED_LIMIT) : tags;

  return (
    <div className="tag-filter">
      <label className="tag-filter-label">Filter by Object Tag:</label>
      <div className="tag-filter-chips">
        {visibleTags.map((tag) => (
          <button
            key={tag}
            type="button"
            className={`tag-chip ${selectedTags.includes(tag) ? "selected" : ""}`}
            onClick={() => handleToggle(tag)}
          >
            {tag}
          </button>
        ))}
        {needsCollapse && (
          <button
            type="button"
            className="tag-chip toggle"
            onClick={() => setExpanded((prev) => !prev)}
          >
            {expanded ? "Show less" : `+${tags.length - COLLAPSED_LIMIT} more`}
          </button>
        )}
        {selectedTags.length > 0 && (
          <button
            type="button"
            className="tag-chip clear"
            onClick={() => onChange([])}
          >
            Clear
          </button>
        )}
      </div>
    </div>
  );
}
