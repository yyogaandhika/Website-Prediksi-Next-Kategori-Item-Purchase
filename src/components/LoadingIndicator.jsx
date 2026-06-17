export default function LoadingIndicator() {
  return (
    <div className="loading" role="status" aria-live="polite">
      <div className="loading__dots">
        <span />
        <span />
        <span />
      </div>
      <p className="loading__text">Menyusuri pola riwayat belanja…</p>
    </div>
  );
}
