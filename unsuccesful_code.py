def two_piece_merger(sides_comparison: SidesComparison):
    image1 = pieces[sides_comparison.side1.fragment_idx].value
    h, w = image1.shape[:2] 
    start_points1 = sides_comparison.side1.side_indexes_of_fragment
    centroid = [pieces[sides_comparison.side1.fragment_idx].cx, pieces[sides_comparison.side1.fragment_idx].cy]
    three_start_points1 = np.array([start_points1[0],centroid,start_points1[-1]],dtype=np.float32)
    three_destination_points1 = np.array([[0, h-1], centroid, [h-1, h-1]],dtype=np.float32)

    print(f"start:{three_start_points1} destination: {three_destination_points1}")
    transformation_matrix = cv.getAffineTransform(three_start_points1, three_destination_points1)
    rotated_image1 = cv.warpAffine(image1, transformation_matrix, (w, h))
    image2 = pieces[sides_comparison.side2.fragment_idx].value
    h, w = image2.shape[:2] 

    start_points2 = sides_comparison.side2.side_indexes_of_fragment
    centroid = [pieces[sides_comparison.side2.fragment_idx].cx, pieces[sides_comparison.side2.fragment_idx].cy]
    three_start_points2 = np.array([start_points1[0],centroid,start_points1[-1]],dtype=np.float32)
    three_destination_points2 = np.array([[h-1, 0], centroid, [0, 0]],dtype=np.float32)

    # print(f"start:{three_start_points1} destination: {three_destination_points}")
    transformation_matrix = cv.getAffineTransform(three_start_points2, three_destination_points2)
    rotated_image2 = cv.warpAffine(image2, transformation_matrix, (w, h))

    plt.figure(figsize=(12, 6)) 

    plt.subplot(1, 2, 1)  
    plt.imshow(rotated_image1) 
    plt.subplot(1, 2, 2)  
    plt.imshow(rotated_image2)

    plt.show()