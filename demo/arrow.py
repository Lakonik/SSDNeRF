import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    # Sample data
    x = [1, 2, 3]
    y = [2, 3, 4]
    z = [3, 4, 5]

    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    # Add arrow
    ax.quiver(1, 2, 3, 1, 1, 1, color='red')

    # Set plot labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Scatter Plot with Arrow')

    # Show plot
    plt.show()

if __name__ == "__main__":
    main()