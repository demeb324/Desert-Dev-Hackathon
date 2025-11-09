/**
 * ErrorAlert Component
 *
 * Displays error messages with optional retry button
 * Styled to match OptiGreen theme
 */

interface ErrorAlertProps {
  error: string | null;
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  error,
  onRetry,
  onDismiss,
  className = '',
}) => {
  if (!error) return null;

  return (
    <div
      className={`bg-red-900/30 border border-red-500/50 rounded-lg p-4 ${className}`}
      role="alert"
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 text-red-400 text-xl">⚠️</div>
        <div className="flex-1">
          <p className="text-red-200 text-sm">{error}</p>
        </div>
        <div className="flex gap-2">
          {onRetry && (
            <button
              onClick={onRetry}
              className="text-cyan-400 hover:text-cyan-300 text-sm font-medium transition-colors"
            >
              Retry
            </button>
          )}
          {onDismiss && (
            <button
              onClick={onDismiss}
              className="text-gray-400 hover:text-gray-300 text-sm transition-colors"
              aria-label="Dismiss error"
            >
              ✕
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
