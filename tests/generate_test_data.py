"""Generate test dataset of random seismic-like events."""

import argparse
import numpy as np
from pathlib import Path


def generate_seismic_like_event(
    n_samples,
    n_traces,
    n_time_samples,
    n_events_min=3,
    n_events_max=8,
    noise_level=0.1,
):
    """
    Generate a 3D numpy array of random seismic-like events.
    
    Parameters
    ----------
    n_samples : int
        Number of samples (shots) in the dataset
    n_traces : int
        Number of traces per sample
    n_time_samples : int
        Number of time samples per trace
    n_events_min : int
        Minimum number of seismic events per sample
    n_events_max : int
        Maximum number of seismic events per sample
    noise_level : float
        Level of random noise to add
        
    Returns
    -------
    data : np.array
        3D array of shape (n_samples, n_time_samples, n_traces)
    """
    data = np.zeros((n_samples, n_time_samples, n_traces))
    
    for sample_idx in range(n_samples):
        # Generate random number of events for this sample
        n_events = np.random.randint(n_events_min, n_events_max + 1)
        
        for _ in range(n_events):
            # Random event parameters
            event_time = np.random.randint(n_time_samples // 4, n_time_samples * 3 // 4)
            event_trace = np.random.randint(0, n_traces)
            event_amplitude = np.random.uniform(-1.0, 1.0)
            event_width_time = np.random.randint(5, 20)
            event_width_trace = np.random.randint(1, 5)
            
            # Create a hyperbolic or linear event shape
            event_type = np.random.choice(['hyperbolic', 'linear', 'point'])
            
            if event_type == 'hyperbolic':
                # Hyperbolic event (like a reflection)
                t0 = event_time
                x0 = event_trace
                v = np.random.uniform(1000, 3000)  # Velocity
                
                for t in range(n_time_samples):
                    for x in range(n_traces):
                        t_calc = np.sqrt(t0**2 + ((x - x0) / v)**2)
                        if abs(t - t_calc) < event_width_time:
                            # Ricker wavelet-like shape
                            tau = (t - t_calc) / event_width_time
                            amplitude = event_amplitude * (1 - 2 * np.pi**2 * tau**2) * np.exp(-np.pi**2 * tau**2)
                            data[sample_idx, t, x] += amplitude
            
            elif event_type == 'linear':
                # Linear dipping event
                dip = np.random.uniform(-0.5, 0.5)
                t0 = event_time
                x0 = event_trace
                
                for t in range(n_time_samples):
                    for x in range(n_traces):
                        t_calc = t0 + dip * (x - x0)
                        if abs(t - t_calc) < event_width_time and abs(x - x0) < event_width_trace * 10:
                            tau = (t - t_calc) / event_width_time
                            amplitude = event_amplitude * np.exp(-tau**2)
                            data[sample_idx, t, x] += amplitude
            
            else:  # point event
                # Point-like event
                for t in range(n_time_samples):
                    for x in range(n_traces):
                        dist_time = abs(t - event_time)
                        dist_trace = abs(x - event_trace)
                        if dist_time < event_width_time and dist_trace < event_width_trace:
                            amplitude = event_amplitude * np.exp(
                                -(dist_time**2 / (2 * event_width_time**2) +
                                  dist_trace**2 / (2 * event_width_trace**2))
                            )
                            data[sample_idx, t, x] += amplitude
        
        # Add random noise
        data[sample_idx] += np.random.normal(0, noise_level, (n_time_samples, n_traces))
        
        # Normalize each sample
        data[sample_idx] = data[sample_idx] / (np.abs(data[sample_idx]).max() + 1e-10) * 0.5
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate test dataset of random seismic-like events")
    parser.add_argument(
        "--output",
        type=str,
        default="test_seismic_data.npy",
        help="Output file path for the generated data",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples (shots) in the dataset",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=64,
        help="Number of traces per sample",
    )
    parser.add_argument(
        "--n-time-samples",
        type=int,
        default=128,
        help="Number of time samples per trace",
    )
    parser.add_argument(
        "--n-events-min",
        type=int,
        default=3,
        help="Minimum number of seismic events per sample",
    )
    parser.add_argument(
        "--n-events-max",
        type=int,
        default=8,
        help="Maximum number of seismic events per sample",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.1,
        help="Level of random noise to add",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate data
    print(f"Generating test dataset with shape ({args.n_samples}, {args.n_time_samples}, {args.n_traces})...")
    data = generate_seismic_like_event(
        n_samples=args.n_samples,
        n_traces=args.n_traces,
        n_time_samples=args.n_time_samples,
        n_events_min=args.n_events_min,
        n_events_max=args.n_events_max,
        noise_level=args.noise_level,
    )

    # Save data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data)
    
    print(f"Generated test dataset saved to {output_path}")
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"Data mean: {data.mean():.4f}, std: {data.std():.4f}")


if __name__ == "__main__":
    main()

