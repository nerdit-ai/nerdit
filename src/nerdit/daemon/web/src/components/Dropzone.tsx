import { useRef, useState, type DragEvent, type ReactNode } from "react";

export interface DropzoneProps {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
  label?: ReactNode;
  hint?: ReactNode;
  accept?: string;
  multiple?: boolean;
  className?: string;
}

export function Dropzone({
  onFiles,
  disabled = false,
  label = "Drop a file here or click to browse",
  hint,
  accept,
  multiple = false,
  className = ""
}: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [hovering, setHovering] = useState(false);

  function handleFiles(list: FileList | null) {
    if (!list || list.length === 0) return;
    onFiles(Array.from(list));
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setHovering(false);
    if (disabled) return;
    handleFiles(event.dataTransfer.files);
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(event) => {
        if ((event.key === "Enter" || event.key === " ") && !disabled) {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setHovering(true);
      }}
      onDragLeave={() => setHovering(false)}
      onDrop={onDrop}
      aria-disabled={disabled}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-card border border-dashed px-6 py-10 text-center text-14 transition ${
        hovering ? "border-primary bg-primary-subtle text-primary" : "border-border bg-surface-hover text-muted-foreground"
      } ${disabled ? "cursor-not-allowed opacity-50" : ""} ${className}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        className="hidden"
        onChange={(event) => handleFiles(event.target.files)}
      />
      <span className="font-medium text-foreground">{label}</span>
      {hint && <span className="mt-1 text-12 text-subtle-foreground">{hint}</span>}
    </div>
  );
}
