#!/usr/bin/env python3
"""
Run the TDM FastAPI server.

Usage:
    python scripts/run_server.py
    python scripts/run_server.py --model gpt2 --port 8000 --mode whitebox
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

from tdm.config import TDMConfig, set_config


def main():
    parser = argparse.ArgumentParser(description="Run TDM server")
    
    # Server options
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Model options
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--mode", type=str, default="whitebox",
                       choices=["whitebox", "blackbox"])
    
    # Detector options
    parser.add_argument("--layers", type=int, nargs="+", default=[-1])
    parser.add_argument("--pooling", type=str, default="last_token")
    parser.add_argument("--n-canaries", type=int, default=4)
    
    # Thresholds
    parser.add_argument("--block-threshold", type=float, default=0.8)
    parser.add_argument("--reroute-threshold", type=float, default=0.6)
    parser.add_argument("--reroute-model", type=str, default=None,
                       help="Model to use for rerouting")
    
    # Paths
    parser.add_argument("--artifacts", type=str, default="./artifacts")
    parser.add_argument("--logs", type=str, default="./logs")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TDMConfig(
        model_name=args.model,
        device=args.device,
        mode=args.mode,
        target_layers=args.layers,
        pooling=args.pooling,
        n_canaries=args.n_canaries,
        block_threshold=args.block_threshold,
        reroute_threshold=args.reroute_threshold,
        reroute_model=args.reroute_model or args.model,
        artifacts_dir=args.artifacts,
        logs_dir=args.logs,
        server_host=args.host,
        server_port=args.port
    )
    
    # Set global config
    set_config(config)
    
    print("=" * 60)
    print("Triangulated Defection Monitor (TDM) Server")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Mode: {config.mode}")
    print(f"Device: {config.device}")
    print(f"Artifacts: {config.artifacts_dir}")
    print(f"Logs: {config.logs_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST /score         - Score a prompt for defection risk")
    print("  POST /generate_safe - Generate safe response with policy")
    print("  GET  /health        - Health check")
    print("  GET  /config        - Current configuration")
    print("=" * 60)
    
    # Run server
    uvicorn.run(
        "tdm.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
