import { useApp } from '../lib/context';
import { CheckCircle, XCircle, Info, X } from 'lucide-react';

const icons = {
  success: CheckCircle,
  error: XCircle,
  info: Info,
};

const colors = {
  success: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  error: 'text-red-400 bg-red-500/10 border-red-500/20',
  info: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
};

export default function ToastContainer() {
  const { toasts, removeToast } = useApp();

  if (toasts.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-[100] flex flex-col gap-2 max-w-sm">
      {toasts.map(toast => {
        const Icon = icons[toast.type];
        return (
          <div
            key={toast.id}
            className={`animate-toast-in glass flex items-center gap-3 px-4 py-3 rounded-xl border shadow-2xl ${colors[toast.type]}`}
          >
            <Icon className="w-5 h-5 shrink-0" />
            <span className="text-sm font-medium flex-1">{toast.message}</span>
            <button
              onClick={() => removeToast(toast.id)}
              className="opacity-60 hover:opacity-100 transition-opacity"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
