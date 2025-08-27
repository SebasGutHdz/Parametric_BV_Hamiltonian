








def visualize_gradient_flow_results(results: dict, figsize: tuple = (15, 10)):
    """
    Visualize the gradient flow results
    
    Args:
        results: Results dictionary from run_gradient_flow
        figsize: Figure size for plots
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Energy decay plot
    axes[0,0].plot(results['energy_history'])
    axes[0,0].set_xlabel('Iteration')
    axes[0,0].set_ylabel('Energy')
    axes[0,0].set_title('Energy Decay')
    axes[0,0].grid(True)
    axes[0,0].set_yscale('log')
    
    # Parameter norm evolution
    axes[0,1].plot(results['param_norms'])
    axes[0,1].set_xlabel('Iteration') 
    axes[0,1].set_ylabel('Parameter Norm')
    axes[0,1].set_title('Parameter Evolution')
    axes[0,1].grid(True)
    
    # Initial vs Final samples
    initial_samples = results['sample_history'][0]
    final_samples = results['sample_history'][-1]
    
    axes[1,0].scatter(initial_samples[:,0], initial_samples[:,1], alpha=0.5, s=1, label='Initial')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    axes[1,0].set_title('Initial Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True)
    axes[1,0].set_aspect('equal')
    
    axes[1,1].scatter(final_samples[:,0], final_samples[:,1], alpha=0.5, s=1, c='red', label='Final')
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('y') 
    axes[1,1].set_title('Final Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True)
    axes[1,1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
