export default function CategoryChain({ categories, showTarget }) {
  if (!categories.length) return null;

  return (
    <div className="chain" aria-hidden="false">
      {categories.map((cat, idx) => (
        <div className="chain__item" key={`${cat}-${idx}`}>
          <span className="chain__chip">{cat}</span>
          <span className="chain__link" />
        </div>
      ))}
      {showTarget && (
        <div className="chain__item chain__item--target">
          <span className="chain__chip chain__chip--target">?</span>
        </div>
      )}
    </div>
  );
}
