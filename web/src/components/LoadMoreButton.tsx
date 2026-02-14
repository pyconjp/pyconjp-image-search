interface Props {
  onClick: () => void;
  isLoading: boolean;
  visible: boolean;
}

export function LoadMoreButton({ onClick, isLoading, visible }: Props) {
  if (!visible) return null;

  return (
    <div className="load-more">
      <button
        type="button"
        onClick={onClick}
        disabled={isLoading}
        className="load-more-button"
      >
        {isLoading ? "Loading..." : "Load More"}
      </button>
    </div>
  );
}
