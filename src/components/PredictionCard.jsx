const RANK_ACCENT = {
  0: { bar: "var(--amber)", glow: "var(--amber-soft)", label: "Tertinggi" },
  1: { bar: "var(--coral)", glow: "var(--coral-soft)", label: "Kuat" },
  2: { bar: "var(--mint)", glow: "var(--mint-soft)", label: "Potensial" },
};

export default function PredictionCard({ rank, category, probability, animate }) {
  const percent = Math.round(probability * 1000) / 10; // satu desimal
  const accent = RANK_ACCENT[rank] ?? RANK_ACCENT[2];

  return (
    <div className="prediction-card" style={{ "--accent-bar": accent.bar, "--accent-glow": accent.glow }}>
      <div className="prediction-card__top">
        <span className="prediction-card__rank">{String(rank + 1).padStart(2, "0")}</span>
        <div className="prediction-card__heading">
          <h3 className="prediction-card__name">{category}</h3>
          <span className="prediction-card__tag">{accent.label}</span>
        </div>
        <span className="prediction-card__percent">{percent}%</span>
      </div>

      <div className="prediction-card__track" role="progressbar" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100}>
        <div
          className={`prediction-card__fill ${animate ? "prediction-card__fill--animate" : ""}`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
