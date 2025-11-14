import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

def crystal_classifier(coordinates, neighbors_indices, angles, angle_threshold=5.0, dbond=2.21, bounds=(-55, 55), marker_radius=1, mode='center', FileNameGlobal=""):
    N = coordinates.shape[0]

    if mode == 'surface':
        dbond -= 2.21  # Adjust threshold

    # Initialize plot
    plt.figure(figsize=(8, 8))
    plt.axis([bounds[0], bounds[1], bounds[0], bounds[1]])
    plt.gca().set_aspect('equal', adjustable='box')

    def find_clusters(coordinates, neighbors_indices, angles, angle_threshold=5.0):
        clusters = []
        visited = set()

        def dfs(particle, cluster, bond_angle):
            stack = [particle]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    cluster.append(current)
                    for neighbor in neighbors_indices[current]:
                        if neighbor < len(angles):
                            neighbor_angles = angles[neighbor]
                            for neighbor_angle in neighbor_angles:
                                if abs(neighbor_angle - bond_angle) <= angle_threshold:
                                    stack.append(neighbor)
                                    break

        for i in range(N):
            if i not in visited:
                for bond_angle in angles[i]:
                    cluster = []
                    dfs(i, cluster, bond_angle)
                    if cluster:
                        clusters.append(cluster)

        return clusters

    clusters = find_clusters(coordinates, neighbors_indices, angles, angle_threshold)

    def classify_angle(angle):
        if 70 <= angle <= 140:
            return "open_link"
        elif 50 <= angle < 70:
            return "closed_link"
        elif 140 < angle <= 180:
            return "stretched_link"
        else:
            return None

    colors = np.full(N, 'k', dtype=str)
    for i, angle_list in enumerate(angles):
        for bond_angle in angle_list:
            if 70 <= bond_angle <= 140:
                colors[i] = 'g'
            elif 50 <= bond_angle < 70:
                colors[i] = 'r'
            elif 140 < bond_angle <= 180:
                colors[i] = 'c'

    for i in range(N):
        circle = Circle((coordinates[i, 0], coordinates[i, 1]), radius=marker_radius, edgecolor='black', facecolor=colors[i], linewidth=1)
        plt.gca().add_patch(circle)

    # Flatten clusters for membership checks
    all_cluster_indices = {particle for cluster in clusters for particle in cluster}

    result_matrix = []
    for i in range(len(coordinates)):
        bond_types = [classify_angle(angle) for angle in angles[i] if classify_angle(angle)]
        dominant_bond_type = max(set(bond_types), key=bond_types.count) if bond_types else "Unknown"

        result_matrix.append({
            "index": i,
            "coordinate": coordinates[i],
            "crystal_type": dominant_bond_type,
            "neighbors_in_crystal": [n for n in neighbors_indices[i] if n in all_cluster_indices],
            "angle": angles[i],
            "nearest_neighbours": neighbors_indices[i],
            "color": colors[i]
        })

    crystal_summary_matrix = []
    for cluster_id, cluster in enumerate(clusters):
        bond_types = []
        for particle in cluster:
            bond_types.extend([classify_angle(angle) for angle in angles[particle] if classify_angle(angle)])

        dominant_bond_type = max(set(bond_types), key=bond_types.count) if bond_types else "Unknown"
        x_coords = [coordinates[i][0] for i in cluster]
        y_coords = [coordinates[i][1] for i in cluster]
        center_coordinate = (np.mean(x_coords), np.mean(y_coords))

        cluster_color = max(set(colors[cluster]), key=list(colors[cluster]).count)

        crystal_summary_matrix.append({
            "cluster_id": cluster_id,
            "crystal_type": dominant_bond_type,
            "center_coordinate": center_coordinate,
            "particle_indexes": cluster,
            "color": cluster_color
        })


    # Calculate percentages of each crystal type
    crystal_types_count = {"open_link": 0, "closed_link": 0, "stretched_link": 0, "Unknown": 0}
    for particle in result_matrix:
        crystal_type = particle["crystal_type"]
        if crystal_type in crystal_types_count:
            crystal_types_count[crystal_type] += 1
        else:
            crystal_types_count["Unknown"] += 1

    # Calculate percentages
    crystal_types_percentage = {k: (v / N) * 100 for k, v in crystal_types_count.items()}

    print("Crystal Type Percentages:")
    for crystal_type, percentage in crystal_types_percentage.items():
        print(f"{crystal_type}: {percentage:.2f}%")


    OpenLinkPercentage = crystal_types_percentage["open_link"]
    ClosedLinkPercentage = crystal_types_percentage["closed_link"]
    StretchedLinkPercentage = crystal_types_percentage["stretched_link"]
    Unknown = crystal_types_percentage["Unknown"]

    # # Add percentages to the plot
    # plt.text(bounds[0] + 5, bounds[1] - 5, f"Open Link: {OpenLinkPercentage:.2f}%", color='g')
    # plt.text(bounds[0] + 5, bounds[1] - 10, f"Closed Link: {ClosedLinkPercentage:.2f}%", color='r')
    # plt.text(bounds[0] + 5, bounds[1] - 15, f"Stretched Link: {StretchedLinkPercentage:.2f}%", color='c')
    # plt.text(bounds[0] + 5, bounds[1] - 20, f"Unknown: {Unknown:.2f}%", color='k')
    #
    # # Add percentages as a subtitle to the plot
    # subtitle = (f"Open Link: {OpenLinkPercentage:.2f}%, "
    #             f"Closed Link: {ClosedLinkPercentage:.2f}%, "
    #             f"Stretched Link: {StretchedLinkPercentage:.2f}%, "
    #             f"Unknown: {Unknown:.2f}%")
    # plt.suptitle(subtitle, fontsize=10, y=0.95)
    #
    # # Add colored percentages as a subtitle to the plot
    # plt.text(0.5, 1.05, f"Open Link: {OpenLinkPercentage:.2f}%", color='g', fontsize=10, ha='center',
    #          transform=plt.gca().transAxes)
    # plt.text(0.5, 1.02, f"Closed Link: {ClosedLinkPercentage:.2f}%", color='r', fontsize=10, ha='center',
    #          transform=plt.gca().transAxes)
    # plt.text(0.5, 0.99, f"Stretched Link: {StretchedLinkPercentage:.2f}%", color='c', fontsize=10, ha='center',
    #          transform=plt.gca().transAxes)
    # plt.text(0.5, 0.96, f"Unknown: {Unknown:.2f}%", color='k', fontsize=10, ha='center', transform=plt.gca().transAxes)

    # Add colored percentages as a single subtitle to the plot


    subtitle = (f"Open Link: {OpenLinkPercentage:.2f}%", f"Closed Link: {ClosedLinkPercentage:.2f}%",
                f"Stretched Link: {StretchedLinkPercentage:.2f}%", f"Unknown: {Unknown:.2f}%")
    colors = ['g', 'r', 'c', 'k']
    for i, text in enumerate(subtitle):
        plt.text(0.1 + i * 0.3, 1.05, text, color=colors[i], fontsize=10, ha='center', transform=plt.gca().transAxes)


    plt.title('Crystal Bond Angle Classification')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    # Generate a unique plot name using the file name
    plot_folder = "Static Folder Run Output Figures"
    plot_name = os.path.join(plot_folder, f"{os.path.splitext(FileNameGlobal)[0]}_plot.jpg")
    plt.savefig(plot_name)

    plt.show()

    return result_matrix, crystal_summary_matrix, OpenLinkPercentage, ClosedLinkPercentage, StretchedLinkPercentage, Unknown
