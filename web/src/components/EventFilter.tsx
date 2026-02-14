interface Props {
  eventNames: string[];
  selectedEvents: string[];
  onChange: (events: string[]) => void;
}

export function EventFilter({ eventNames, selectedEvents, onChange }: Props) {
  const handleToggle = (name: string) => {
    if (selectedEvents.includes(name)) {
      onChange(selectedEvents.filter((e) => e !== name));
    } else {
      onChange([...selectedEvents, name]);
    }
  };

  if (eventNames.length === 0) return null;

  return (
    <div className="event-filter">
      <label className="event-filter-label">Filter by Event:</label>
      <div className="event-filter-chips">
        {eventNames.map((name) => (
          <button
            key={name}
            type="button"
            className={`event-chip ${selectedEvents.includes(name) ? "selected" : ""}`}
            onClick={() => handleToggle(name)}
          >
            {name}
          </button>
        ))}
        {selectedEvents.length > 0 && (
          <button
            type="button"
            className="event-chip clear"
            onClick={() => onChange([])}
          >
            Clear
          </button>
        )}
      </div>
    </div>
  );
}
