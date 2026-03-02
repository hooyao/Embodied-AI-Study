"""Verify all RL training dependencies are installed correctly.

Run inside the Isaac Sim container:
    /isaac-sim/python.sh /tmp/verify_install.py
"""

import sys

results = []

def check(name, import_fn):
    """Import a module and record success/failure with optional detail."""
    try:
        detail = import_fn()
        results.append((name, "OK", detail or ""))
    except Exception as e:
        results.append((name, "FAIL", str(e)))

print("=== Installation Verification ===")

check("isaaclab",    lambda: __import__("isaaclab") and None)
check("rsl_rl",      lambda: __import__("rsl_rl") and None)
check("skrl",        lambda: f"v{__import__('skrl').__version__}")
check("torch",       lambda: (
    t := __import__("torch"),
    f"v{t.__version__}, CUDA={t.cuda.is_available()}"
    + (f", GPU={t.cuda.get_device_name(0)}" if t.cuda.is_available() else "")
)[-1])
check("gymnasium",   lambda: f"v{__import__('gymnasium').__version__}")
check("tensorboard", lambda: __import__("tensorboard") and None)

# Print results
all_ok = True
for name, status, detail in results:
    icon = "OK" if status == "OK" else "FAIL"
    extra = f" ({detail})" if detail else ""
    print(f"  {name}: {icon}{extra}")
    if status != "OK":
        all_ok = False

if all_ok:
    print("\nAll dependencies verified!")
    sys.exit(0)
else:
    print("\nSome dependencies are missing. See above.")
    sys.exit(1)
