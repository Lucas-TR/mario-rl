# show_stack.py
import matplotlib.pyplot as plt
from mario_env import make_env

if __name__ == "__main__":
    env = make_env(stack_frames=4, grayscale=True, vectorized=True)
    state = env.reset()                  # state: (1, H, W, 4)
    state, reward, done, info = env.step([5])  # acción ejemplo
    plt.figure(figsize=(20, 16))
    for idx in range(state.shape[3]):
        plt.subplot(1, 4, idx + 1)
        plt.imshow(state[0][:, :, idx], cmap="gray")
        plt.axis("off")
        plt.title(f"Frame {idx+1}")
    plt.tight_layout()
    plt.show()
    env.close()
