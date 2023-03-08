
import os
from PIL import Image
from collections import defaultdict
import numpy as np
import argparse

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_name', type=str, default='./sub/', help='original image')
    parser.add_argument('--column_num', type=int, default=2, help='column_num')
    parser.add_argument('--row_num', type=int, default=2, help='row_num')
    parser.add_argument('--output_filename', type=str, default='merged_image', help='filename')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):

    def image_similarity(image1, image2, side1, side2):
        # Convert the images to grayscale arrays
        array1 = np.asarray(image1.convert('L'))
        array2 = np.asarray(image2.convert('L'))

        if side1 == 'left':
            array1 = array1[:, :4]
        elif side1 == 'right':
            array1 = array1[:, -4:]

        if side2 == 'left':
            array2 = array2[:, :4]
        elif side2 == 'right':
            array2 = array2[:, -4:]

        # Compute the mean squared error (MSE) between the two arrays
        mse = np.mean((array1 - array2) ** 2)

        # Compute the peak signal-to-noise ratio (PSNR) between the two arrays
        psnr = 10 * np.log10((255 ** 2) / mse)

        # Compute the structural similarity index (SSIM) between the two arrays
        ssim = 0 # TODO: Implement SSIM computation

        # Compute the average of the three scores
        score = (psnr + ssim) / 2
        return score

    # Define a function to rotate, flip, or leave an image unchanged based on a flag
    def transform_image(image, rotation, horizontal_flip, vertical_flip):
        if rotation != 0:
            image = image.rotate(rotation,expand = 1)
        if horizontal_flip:
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
        if vertical_flip:
            image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
        return image
    


    # Start with 2x2 image
    if opt.column_num == 2 and opt.row_num == 2:

        path = opt.input_file_name
        file_list = os.listdir(path)
        minnum = 2


        # Load image
        image1 = Image.open(path + file_list.pop(0))
        

        # Get the size of the images
        width, height = image1.size
        answersheet = [[0 for __ in range(opt.column_num)] for _ in range(opt.row_num)]
        


        # Consider code that width == height or width != height
        # Need to care dimension if width != height, so make code different
        if width == height:
            score_dict = defaultdict(int)
            pair_dict = defaultdict(list)
            for k in range(len(file_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = Image.open(path + file_list[k])

                # Try all possible combinations of rotations and flips for the two images
                for rotation1 in [0, 270]:
                    for rotation2 in [0, 270]:
                        for horizontal_flip1 in [False, True]:
                            for horizontal_flip2 in [False, True]:
                                for vertical_flip1 in [False, True]:
                                    for vertical_flip2 in [False, True]:
                                        # Transform the two images based on the current flags
                                        transformed_image1 = transform_image(image1, rotation1, horizontal_flip1, vertical_flip1)
                                        transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                                        # Compute the similarity score between the two transformed images
                                        score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                        # If the score is better than the previous best score, update the best pair
                                        if score > best_score:
                                            best_pair = (transformed_image1, transformed_image2)
                                            best_score = score
                score_dict[k] = best_score
                pair_dict[k] = best_pair
            a = [k for k,v in score_dict.items() if v == max(score_dict.values())]

            answersheet[0][0] = pair_dict[a[0]][0]
            answersheet[0][1] = pair_dict[a[0]][1]
            file_list.pop(a[0])

            image1_top = transform_image(answersheet[0][0],90,False,False)
            image1_bottom = transform_image(answersheet[0][0],270,False,False)

            score_dict_top = defaultdict(int)
            pair_dict_top = defaultdict(list)
            score_dict_bottom = defaultdict(int)
            pair_dict_bottom = defaultdict(list)

            for k in range(len(file_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = Image.open(path + file_list[k])

                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [0, 270]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1_top, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
                score_dict_top[k] = best_score
                pair_dict_top[k] = best_pair
            
            for k in range(len(file_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = Image.open(path + file_list[k])

                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [0, 270]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1_bottom, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
                score_dict_bottom[k] = best_score
                pair_dict_bottom[k] = best_pair
            if max(score_dict_top.values()) < max(score_dict_bottom.values()):
                answersheet[0][0] = transform_image(answersheet[0][0],0,False,True)
                answersheet[0][1] = transform_image(answersheet[0][1],0,False,True)
                a = [k for k,v in score_dict_bottom.items() if v == max(score_dict_bottom.values())]
                file_list.pop(a[0])
                answersheet[1][0] = transform_image(pair_dict_bottom[a[0]][1],90,False,True)
            else:
                a = [k for k,v in score_dict_top.items() if v == max(score_dict_top.values())]
                answersheet[1][0] = transform_image(pair_dict_top[a[0]][1],270,False,False)
                file_list.pop(a[0])
            
            image1 = answersheet[1][0]

            for k in range(len(file_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = Image.open(path + file_list[k])

                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [0, 270]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
            answersheet[1][1] = best_pair[1]

            # Create a new blank image with twice the width and the same height
            merged_image = Image.new("RGB", (width * 2, height * 2))

            # Paste the best pair of images on the left and right side of the merged image
            merged_image.paste(answersheet[0][0], (0, 0))
            merged_image.paste(answersheet[0][1], (width, 0))
            merged_image.paste(answersheet[1][0], (0, height))
            merged_image.paste(answersheet[1][1], (width, height))

            # Save the merged image
            merged_image.save(f"{opt.output_filename}.jpg")


        else:
            image_list = []
            
            for k in range(len(file_list)):
                image_tmp = Image.open(path + file_list[k])
                if image_tmp.size[0] != width:
                    image_tmp = transform_image(image_tmp,270,False,False)
                image_list.append(image_tmp)
                 
            score_dict = defaultdict(int)
            pair_dict = defaultdict(list)
            for k in range(len(image_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = image_list[k]

                # Try all possible combinations of rotations and flips for the two images

                for horizontal_flip1 in [False, True]:
                    for horizontal_flip2 in [False, True]:
                        for vertical_flip1 in [False, True]:
                            for vertical_flip2 in [False, True]:
                                # Transform the two images based on the current flags
                                transformed_image1 = transform_image(image1, 0, horizontal_flip1, vertical_flip1)
                                transformed_image2 = transform_image(image2, 0, horizontal_flip2, vertical_flip2)

                                # Compute the similarity score between the two transformed images
                                score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                # If the score is better than the previous best score, update the best pair
                                if score > best_score:
                                    best_pair = (transformed_image1, transformed_image2)
                                    best_score = score
                score_dict[k] = best_score
                pair_dict[k] = best_pair
            a = [k for k,v in score_dict.items() if v == max(score_dict.values())]
            # image1 = pair_dict[a[0]][0]
            answersheet[0][0] = pair_dict[a[0]][0]
            answersheet[0][1] = pair_dict[a[0]][1]
            image_list.pop(a[0])

            image1_top = transform_image(answersheet[0][0],90,False,False)
            image1_bottom = transform_image(answersheet[0][0],270,False,False)

            score_dict_top = defaultdict(int)
            pair_dict_top = defaultdict(list)
            score_dict_bottom = defaultdict(int)
            pair_dict_bottom = defaultdict(list)

            for k in range(len(image_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = image_list[k]

                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [90]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1_top, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
                score_dict_top[k] = best_score
                pair_dict_top[k] = best_pair
            
            for k in range(len(image_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = image_list[k]

                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [90]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1_bottom, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            # score1 = image_similarity(transformed_image1, transformed_image2, 'left', 'right')
                            # score2 = image_similarity(transformed_image1, transformed_image2, 'right', 'left')
                            # score = max(score1, score2)
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
                score_dict_bottom[k] = best_score
                pair_dict_bottom[k] = best_pair
            if max(score_dict_top.values()) < max(score_dict_bottom.values()):
                answersheet[0][0] = transform_image(answersheet[0][0],0,False,True)
                answersheet[0][1] = transform_image(answersheet[0][1],0,False,True)
                a = [k for k,v in score_dict_bottom.items() if v == max(score_dict_bottom.values())]
                image_list.pop(a[0])
                answersheet[1][0] = transform_image(pair_dict_bottom[a[0]][1],90,False,True)
            else:
                a = [k for k,v in score_dict_top.items() if v == max(score_dict_top.values())]
                answersheet[1][0] = transform_image(pair_dict_top[a[0]][1],270,False,False)
                image_list.pop(a[0])
            
            image1 = answersheet[1][0]

            for k in range(len(image_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = image_list[k]
                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [0]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
            answersheet[1][1] = best_pair[1]

            # Create a new blank image with twice the width and the same height
            merged_image = Image.new("RGB", (width * 2, height * 2))

            # Paste the best pair of images on the left and right side of the merged image
            merged_image.paste(answersheet[0][0], (0, 0))
            merged_image.paste(answersheet[0][1], (width, 0))
            merged_image.paste(answersheet[1][0], (0, height))
            merged_image.paste(answersheet[1][1], (width, height))

            # Save the merged image
            merged_image.save(f"{opt.output_filename}.jpg")









        
    else:
        path = opt.input_file_name
        file_list = os.listdir(path)

        # Load the two images
        image1 = Image.open(path + file_list[0])
        minnum = min(opt.column_num,opt.row_num)

        # Get the size of the images
        width, height = image1.size
        answersheet = [[0 for __ in range(minnum)] for _ in range(minnum)]
        image_list = []
        
            
        
        for k in range(len(file_list)):
            image_tmp = Image.open(path + file_list[k])
            if image_tmp.size[0] != width:
                image_tmp = transform_image(image_tmp,270,False,False)
            image_list.append(image_tmp)


        # Consider code that width == height or width != height
        # Need to care dimension if width != height, so make code different
        if width != height:
        

            score_dict = defaultdict(int)
            board = [[0 for _ in range(4)] for __ in range(len(image_list))]
            for k in range(len(image_list)):
                total_best_score = 0
                image1 = image_list[k]
                for t in range(len(image_list)):
                    if k != t:
                        
                        
                    # Define variables to store the best pair of images and their similarity score
                        best_score = 0
                        image2 = image_list[t]

                        # Try all possible combinations of rotations and flips for the two images

                        
                        for horizontal_flip2 in [False, True]:
                            for vertical_flip2 in [False, True]:
                                # Transform the two images based on the current flags
                                transformed_image1 = transform_image(image1, 0, False, False)
                                transformed_image2 = transform_image(image2, 0, horizontal_flip2, vertical_flip2)

                                # Compute the similarity score between the two transformed images
                                score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                # If the score is better than the previous best score, update the best pair
                                if score > best_score:
                                    best_pair = (transformed_image1, transformed_image2)
                                    best_score = score

                        if best_score>total_best_score:
                            total_best_score = best_score
                board[k][0] = total_best_score
                score_dict[k] += total_best_score

                total_best_score = 0
                image1_horizontal = transform_image(image1,0,True,False)
                for t in range(len(image_list)):
                    if k != t:
                        
                        
                    # Define variables to store the best pair of images and their similarity score
                        best_score = 0
                        image2 = image_list[t]

                        # Try all possible combinations of rotations and flips for the two images

                        
                        for horizontal_flip2 in [False, True]:
                            for vertical_flip2 in [False, True]:
                                # Transform the two images based on the current flags
                                transformed_image1 = transform_image(image1_horizontal, 0, False, False)
                                transformed_image2 = transform_image(image2, 0, horizontal_flip2, vertical_flip2)

                                # Compute the similarity score between the two transformed images
                                score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                # If the score is better than the previous best score, update the best pair
                                if score > best_score:
                                    best_pair = (transformed_image1, transformed_image2)
                                    best_score = score

                        if best_score>total_best_score:
                            total_best_score = best_score
                board[k][1] = total_best_score
                score_dict[k] += total_best_score

                total_best_score = 0
                image1_rotate = transform_image(image1,90,False,False)
                for t in range(len(image_list)):
                    if k != t:
                        
                        
                    # Define variables to store the best pair of images and their similarity score
                        best_score = 0
                        image2 = image_list[t]

                        # Try all possible combinations of rotations and flips for the two images

                        
                        for horizontal_flip2 in [False, True]:
                            for vertical_flip2 in [False, True]:
                                # Transform the two images based on the current flags
                                transformed_image1 = transform_image(image1_rotate, 0, False, False)
                                transformed_image2 = transform_image(image2, 90, horizontal_flip2, vertical_flip2)

                                # Compute the similarity score between the two transformed images
                                score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                # If the score is better than the previous best score, update the best pair
                                if score > best_score:
                                    best_pair = (transformed_image1, transformed_image2)
                                    best_score = score

                        if best_score>total_best_score:
                            total_best_score = best_score
                board[k][2] = total_best_score
                score_dict[k] += total_best_score

                total_best_score = 0
                image1_rotate_horizontal = transform_image(image1,90,True,False)
                for t in range(len(image_list)):
                    if k != t:
                        
                        
                    # Define variables to store the best pair of images and their similarity score
                        best_score = 0
                        image2 = image_list[t]

                        # Try all possible combinations of rotations and flips for the two images

                        
                        for horizontal_flip2 in [False, True]:
                            for vertical_flip2 in [False, True]:
                                # Transform the two images based on the current flags
                                transformed_image1 = transform_image(image1_rotate_horizontal, 0, False, False)
                                transformed_image2 = transform_image(image2, 90, horizontal_flip2, vertical_flip2)

                                # Compute the similarity score between the two transformed images
                                score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                # If the score is better than the previous best score, update the best pair
                                if score > best_score:
                                    best_pair = (transformed_image1, transformed_image2)
                                    best_score = score

                        if best_score>total_best_score:
                            total_best_score = best_score
                board[k][3] = total_best_score
                score_dict[k] += total_best_score

            a = [k for k,v in score_dict.items() if v == min(score_dict.values())]
            max_index = board[a[0]].index(max(board[a[0]]))
            board[a[0]][max_index] = 0
            next_max_index = board[a[0]].index(max(board[a[0]]))
            checkside = [max_index, next_max_index]
            start_image = image_list.pop(a[0])
            if 1 in checkside:
                start_image = transform_image(start_image, 0, True, False)
            if 3 in checkside:
                start_image = transform_image(start_image, 0, False, True)
            answersheet[0][0] = start_image
            for i in range(minnum-1):
                image1 = answersheet[0][i]
                popnum = 0
                total_best_score = 0
                total_best_pair = (0,0)
                total_popnum = 0

                for k in range(len(image_list)):
                    
                    # Define variables to store the best pair of images and their similarity score
                    best_pair = (None, None)
                    best_score = 0
                    image2 = image_list[k]
                    
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1, 0, False, False)
                            transformed_image2 = transform_image(image2, 0, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
                                popnum = k
                    if best_score > total_best_score:

                        total_best_score = best_score
                        total_best_pair = best_pair
                        total_popnum = popnum

                answersheet[0][i+1] = total_best_pair[1]
                image_list.pop(total_popnum)

            image1_top = transform_image(answersheet[0][0],90,False,False)
            image1_bottom = transform_image(answersheet[0][0],270,False,False)

            score_dict_top = defaultdict(int)
            pair_dict_top = defaultdict(list)
            score_dict_bottom = defaultdict(int)
            pair_dict_bottom = defaultdict(list)

            for k in range(len(image_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = image_list[k]

                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [90]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1_top, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
                score_dict_top[k] = best_score
                pair_dict_top[k] = best_pair
            
            for k in range(len(image_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = image_list[k]

                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [90]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1_bottom, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
                score_dict_bottom[k] = best_score
                pair_dict_bottom[k] = best_pair
            if max(score_dict_top.values()) < max(score_dict_bottom.values()):
                for i in range(len(answersheet[0])):
                    answersheet[0][i] = transform_image(answersheet[0][i],0,False,True)
                a = [k for k,v in score_dict_bottom.items() if v == max(score_dict_bottom.values())]
                image_list.pop(a[0])
                answersheet[1][0] = transform_image(pair_dict_bottom[a[0]][1],90,False,True)
            else:
                a = [k for k,v in score_dict_top.items() if v == max(score_dict_top.values())]
                answersheet[1][0] = transform_image(pair_dict_top[a[0]][1],270,False,False)
                image_list.pop(a[0])

            image1_top = transform_image(answersheet[1][0],90,False,False)
            
            popnum = 0
            total_best_score = 0
            total_best_pair = (0,0)
            total_popnum = 0
            for k in range(len(image_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                
                image2 = image_list[k]

                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [90]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1_top, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
                                popnum = k

                if best_score > total_best_score:

                    total_best_score = best_score
                    total_best_pair = best_pair
                    total_popnum = popnum
            answersheet[2][0] = transform_image(total_best_pair[1],270,False,False)
            image_list.pop(total_popnum)

            for i in range(min(opt.column_num,opt.row_num)-1):
                for j in range(1,min(opt.column_num,opt.row_num)):
                    image1 = answersheet[j][i]
                    popnum = 0
                    total_best_score = 0
                    total_best_pair = (0,0)
                    total_popnum = 0

                    for k in range(len(image_list)):
                    # Define variables to store the best pair of images and their similarity score
                        best_pair = (None, None)
                        best_score = 0
                        image2 = image_list[k]
                        # Try all possible combinations of rotations and flips for the two images
                        
                        
                        for horizontal_flip2 in [False, True]:
                            
                            for vertical_flip2 in [False, True]:
                                # Transform the two images based on the current flags
                                transformed_image1 = transform_image(image1, 0, False, False)
                                transformed_image2 = transform_image(image2, 0, horizontal_flip2, vertical_flip2)

                                # Compute the similarity score between the two transformed images
                                score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                # If the score is better than the previous best score, update the best pair
                                if score > best_score:
                                    best_pair = (transformed_image1, transformed_image2)
                                    best_score = score
                                    popnum = k

                        if best_score > total_best_score:

                            total_best_score = best_score
                            total_best_pair = best_pair
                            total_popnum = popnum

                
                    answersheet[j][i+1] = total_best_pair[1]
                    image_list.pop(total_popnum)
            if opt.column_num != opt.row_num:
                minnum = min(opt.column_num, opt.row_num)
                test_image = answersheet[minnum-1][minnum-1]

                total_best_score = 0
                total_best_pair = (0,0)
                total_popnum = 0
                for k in range(len(image_list)):
                # Define variables to store the best pair of images and their similarity score
                    best_pair = (None, None)
                    best_score = 0
                    
                    image2 = image_list[k]

                    # Try all possible combinations of rotations and flips for the two images
                    
                    for rotation2 in [0]:
                        
                        for horizontal_flip2 in [False, True]:
                            
                            for vertical_flip2 in [False, True]:
                                # Transform the two images based on the current flags
                                transformed_image1 = transform_image(test_image, 0, False, False)
                                transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                                # Compute the similarity score between the two transformed images
                                score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                # If the score is better than the previous best score, update the best pair
                                if score > best_score:
                                    best_pair = (transformed_image1, transformed_image2)
                                    best_score = score
                                    popnum = k

                    if best_score > total_best_score:

                        total_best_score = best_score
                        total_best_pair = best_pair
                        total_popnum = popnum



                test_image_rotate = transform_image(test_image,90,False,False)

                total_best_score_rotate = 0
                total_best_pair_rotate = (0,0)
                total_popnum_rotate = 0
                for k in range(len(image_list)):
                # Define variables to store the best pair of images and their similarity score
                    best_pair = (None, None)
                    best_score = 0
                    image2 = image_list[k]

                    # Try all possible combinations of rotations and flips for the two images
                    
                    for rotation2 in [90]:
                        
                        for horizontal_flip2 in [False, True]:
                            
                            for vertical_flip2 in [False, True]:
                                # Transform the two images based on the current flags
                                transformed_image1 = transform_image(test_image_rotate, 0, False, False)
                                transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                                # Compute the similarity score between the two transformed images
                                score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                # If the score is better than the previous best score, update the best pair
                                if score > best_score:
                                    best_pair = (transformed_image1, transformed_image2)
                                    best_score = score
                                    popnum = k

                    if best_score > total_best_score:

                        total_best_score_rotate = best_score
                        total_best_pair_rotate = best_pair
                        total_popnum_rotate = popnum

                if total_best_score > total_best_score_rotate:
                    image_list.pop(total_popnum)
                    answersheet[minnum-1].append(total_best_pair[1])
                    for fill in range(minnum-1):
                        image1 = answersheet[fill][minnum-1]
                        popnum = 0
                        total_best_score = 0
                        total_best_pair = (0,0)
                        total_popnum = 0

                        for k in range(len(image_list)):
                        # Define variables to store the best pair of images and their similarity score
                            best_pair = (None, None)
                            best_score = 0
                            image2 = image_list[k]
                            # Try all possible combinations of rotations and flips for the two images
                            
                            
                            for horizontal_flip2 in [False, True]:
                                
                                for vertical_flip2 in [False, True]:
                                    # Transform the two images based on the current flags
                                    transformed_image1 = transform_image(image1, 0, False, False)
                                    transformed_image2 = transform_image(image2, 0, horizontal_flip2, vertical_flip2)

                                    # Compute the similarity score between the two transformed images
                                    score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                    # If the score is better than the previous best score, update the best pair
                                    if score > best_score:
                                        best_pair = (transformed_image1, transformed_image2)
                                        best_score = score
                                        popnum = k

                            if best_score > total_best_score:

                                total_best_score = best_score
                                total_best_pair = best_pair
                                total_popnum = popnum

                    
                        answersheet[fill].append(total_best_pair[1])
                        image_list.pop(total_popnum)
                else:
                    image_list.pop(total_popnum_rotate)
                    answersheet.append([0,0,0])
                    answersheet[minnum][minnum-1] = transform_image(total_best_pair_rotate[1],270,False,False)
                    for fill in range(minnum-1):
                        image1 = transform_image(answersheet[minnum][minnum-1-fill],0,True,False)
                        popnum = 0
                        total_best_score = 0
                        total_best_pair = (0,0)
                        total_popnum = 0

                        for k in range(len(image_list)):
                        # Define variables to store the best pair of images and their similarity score
                            best_pair = (None, None)
                            best_score = 0
                            image2 = image_list[k]
                            # Try all possible combinations of rotations and flips for the two images
                            
                            
                            for horizontal_flip2 in [False, True]:
                                
                                for vertical_flip2 in [False, True]:
                                    # Transform the two images based on the current flags
                                    transformed_image1 = transform_image(image1, 0, False, False)
                                    transformed_image2 = transform_image(image2, 0, horizontal_flip2, vertical_flip2)

                                    # Compute the similarity score between the two transformed images
                                    score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                    # If the score is better than the previous best score, update the best pair
                                    if score > best_score:
                                        best_pair = (transformed_image1, transformed_image2)
                                        best_score = score
                                        popnum = k

                            if best_score > total_best_score:

                                total_best_score = best_score
                                total_best_pair = best_pair
                                total_popnum = popnum

                        answersheet[minnum][minnum-1-fill-1] = transform_image(total_best_pair[1],0,True,False)
                        image_list.pop(total_popnum)


            merged_image = Image.new("RGB", (width * len(answersheet[0]), height * len(answersheet)))

            # Paste the best pair of images on the left and right side of the merged image
            for i in range(len(answersheet)):
                for j in range(len(answersheet[0])):
                    merged_image.paste(answersheet[i][j], (j*width,i*height))

            # Save the merged image
            merged_image.save(f"{opt.output_filename}.jpg")




        else:
            score_dict = defaultdict(int)
            board = [[0 for _ in range(4)] for __ in range(len(image_list))]
            for k in range(len(image_list)):
                total_best_score = 0
                image1 = image_list[k]
                for t in range(len(image_list)):
                    if k != t:
                        
                        
                    # Define variables to store the best pair of images and their similarity score
                        best_score = 0
                        image2 = image_list[t]

                        # Try all possible combinations of rotations and flips for the two images

                        for rotation2 in [0,270]:
                            for horizontal_flip2 in [False, True]:
                                for vertical_flip2 in [False, True]:
                                    # Transform the two images based on the current flags
                                    transformed_image1 = transform_image(image1, 0, False, False)
                                    transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                                    # Compute the similarity score between the two transformed images
                                    score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                    # If the score is better than the previous best score, update the best pair
                                    if score > best_score:
                                        best_pair = (transformed_image1, transformed_image2)
                                        best_score = score
                        # total_best_score += best_score

                        if best_score>total_best_score:
                            total_best_score = best_score
                board[k][0] = total_best_score
                score_dict[k] += total_best_score

                total_best_score = 0
                image1_horizontal = transform_image(image1,0,True,False)
                for t in range(len(image_list)):
                    if k != t:
                        
                        
                    # Define variables to store the best pair of images and their similarity score
                        best_score = 0
                        image2 = image_list[t]

                        # Try all possible combinations of rotations and flips for the two images

                        for rotation2 in [0,270]:
                            for horizontal_flip2 in [False, True]:
                                for vertical_flip2 in [False, True]:
                                    # Transform the two images based on the current flags
                                    transformed_image1 = transform_image(image1_horizontal, 0, False, False)
                                    transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                                    # Compute the similarity score between the two transformed images
                                    score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                    # If the score is better than the previous best score, update the best pair
                                    if score > best_score:
                                        best_pair = (transformed_image1, transformed_image2)
                                        best_score = score

                        if best_score>total_best_score:
                            total_best_score = best_score
                board[k][1] = total_best_score
                score_dict[k] += total_best_score

                total_best_score = 0
                image1_rotate = transform_image(image1,90,False,False)
                for t in range(len(image_list)):
                    if k != t:
                        
                        
                    # Define variables to store the best pair of images and their similarity score
                        best_score = 0
                        image2 = image_list[t]

                        # Try all possible combinations of rotations and flips for the two images

                        for rotation2 in [0,270]:
                            for horizontal_flip2 in [False, True]:
                                for vertical_flip2 in [False, True]:
                                    # Transform the two images based on the current flags
                                    transformed_image1 = transform_image(image1_rotate, 0, False, False)
                                    transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                                    # Compute the similarity score between the two transformed images
                                    score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                    # If the score is better than the previous best score, update the best pair
                                    if score > best_score:
                                        best_pair = (transformed_image1, transformed_image2)
                                        best_score = score
                       

                        if best_score>total_best_score:
                            total_best_score = best_score
                board[k][2] = total_best_score
                score_dict[k] += total_best_score

                total_best_score = 0
                image1_rotate_horizontal = transform_image(image1,90,True,False)
                for t in range(len(image_list)):
                    if k != t:
                        
                        
                    # Define variables to store the best pair of images and their similarity score
                        best_score = 0
                        image2 = image_list[t]

                        # Try all possible combinations of rotations and flips for the two images

                        for rotation2 in [0,270]:
                            for horizontal_flip2 in [False, True]:
                                for vertical_flip2 in [False, True]:
                                    # Transform the two images based on the current flags
                                    transformed_image1 = transform_image(image1_rotate_horizontal, 0, False, False)
                                    transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                                    # Compute the similarity score between the two transformed images
                                    score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                    # If the score is better than the previous best score, update the best pair
                                    if score > best_score:
                                        best_pair = (transformed_image1, transformed_image2)
                                        best_score = score
                  

                        if best_score>total_best_score:
                            total_best_score = best_score
                board[k][3] = total_best_score
                score_dict[k] += total_best_score


            a = [k for k,v in score_dict.items() if v == min(score_dict.values())]
            max_index = board[a[0]].index(max(board[a[0]]))
            board[a[0]][max_index] = 0
            next_max_index = board[a[0]].index(max(board[a[0]]))
            checkside = [max_index, next_max_index]
            start_image = image_list.pop(a[0])
            if 1 in checkside:
                start_image = transform_image(start_image, 0, True, False)
            if 3 in checkside:
                start_image = transform_image(start_image, 0, False, True)
            answersheet[0][0] = start_image
            for i in range(minnum-1):
                image1 = answersheet[0][i]
                popnum = 0
                total_best_score = 0
                total_best_pair = (0,0)
                total_popnum = 0

                for k in range(len(image_list)):
                    
                    # Define variables to store the best pair of images and their similarity score
                    best_pair = (None, None)
                    best_score = 0
                    image2 = image_list[k]
                    # Try all possible combinations of rotations and flips for the two images
                    
                    for rotation2 in [0,270]:
                        for horizontal_flip2 in [False, True]:
                            
                            for vertical_flip2 in [False, True]:
                                # Transform the two images based on the current flags
                                transformed_image1 = transform_image(image1, 0, False, False)
                                transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                                # Compute the similarity score between the two transformed images
                                score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                # If the score is better than the previous best score, update the best pair
                                if score > best_score:
                                    best_pair = (transformed_image1, transformed_image2)
                                    best_score = score
                                    popnum = k
                    if best_score > total_best_score:

                        total_best_score = best_score
                        total_best_pair = best_pair
                        total_popnum = popnum

                answersheet[0][i+1] = total_best_pair[1]
                image_list.pop(total_popnum)

            image1_top = transform_image(answersheet[0][0],90,False,False)
            image1_bottom = transform_image(answersheet[0][0],270,False,False)

            score_dict_top = defaultdict(int)
            pair_dict_top = defaultdict(list)
            score_dict_bottom = defaultdict(int)
            pair_dict_bottom = defaultdict(list)

            for k in range(len(image_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = image_list[k]

                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [0, 90, 180, 270]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1_top, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
                score_dict_top[k] = best_score
                pair_dict_top[k] = best_pair
            
            for k in range(len(image_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                image2 = image_list[k]

                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [0, 90, 180, 270]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1_bottom, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
                score_dict_bottom[k] = best_score
                pair_dict_bottom[k] = best_pair
            if max(score_dict_top.values()) < max(score_dict_bottom.values()):
                for i in range(len(answersheet[0])):
                    answersheet[0][i] = transform_image(answersheet[0][i],0,False,True)
                a = [k for k,v in score_dict_bottom.items() if v == max(score_dict_bottom.values())]
                image_list.pop(a[0])
                answersheet[1][0] = transform_image(pair_dict_bottom[a[0]][1],90,False,True)
            else:
                a = [k for k,v in score_dict_top.items() if v == max(score_dict_top.values())]
                answersheet[1][0] = transform_image(pair_dict_top[a[0]][1],270,False,False)
                image_list.pop(a[0])

            image1_top = transform_image(answersheet[1][0],90,False,False)
            
            popnum = 0
            total_best_score = 0
            total_best_pair = (0,0)
            total_popnum = 0
            for k in range(len(image_list)):
            # Define variables to store the best pair of images and their similarity score
                best_pair = (None, None)
                best_score = 0
                
                image2 = image_list[k]

                # Try all possible combinations of rotations and flips for the two images
                
                for rotation2 in [0, 90, 180, 270]:
                    
                    for horizontal_flip2 in [False, True]:
                        
                        for vertical_flip2 in [False, True]:
                            # Transform the two images based on the current flags
                            transformed_image1 = transform_image(image1_top, 0, False, False)
                            transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                            # Compute the similarity score between the two transformed images
                            score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                            # If the score is better than the previous best score, update the best pair
                            if score > best_score:
                                best_pair = (transformed_image1, transformed_image2)
                                best_score = score
                                popnum = k

                if best_score > total_best_score:

                    total_best_score = best_score
                    total_best_pair = best_pair
                    total_popnum = popnum
            answersheet[2][0] = transform_image(total_best_pair[1],270,False,False)
            image_list.pop(total_popnum)

            for i in range(minnum-1):
                # for j in range(1,opt.row_num):
                for j in range(1,minnum):
                    image1 = answersheet[j][i]
                    popnum = 0
                    total_best_score = 0
                    total_best_pair = (0,0)
                    total_popnum = 0

                    for k in range(len(image_list)):
                    # Define variables to store the best pair of images and their similarity score
                        best_pair = (None, None)
                        best_score = 0
                        image2 = image_list[k]
                        # Try all possible combinations of rotations and flips for the two images
                        
                        for rotation2 in [0, 90, 180, 270]:
                            for horizontal_flip2 in [False, True]:
                                
                                for vertical_flip2 in [False, True]:
                                    # Transform the two images based on the current flags
                                    transformed_image1 = transform_image(image1, 0, False, False)
                                    transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                                    # Compute the similarity score between the two transformed images
                                    score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                    # If the score is better than the previous best score, update the best pair
                                    if score > best_score:
                                        best_pair = (transformed_image1, transformed_image2)
                                        best_score = score
                                        popnum = k

                        if best_score > total_best_score:

                            total_best_score = best_score
                            total_best_pair = best_pair
                            total_popnum = popnum

                    answersheet[j][i+1] = total_best_pair[1]
                    image_list.pop(total_popnum)

            if opt.column_num != opt.row_num:
                minnum = min(opt.column_num, opt.row_num)
                test_image = answersheet[minnum-1][minnum-1]

                total_best_score = 0
                total_best_pair = (0,0)
                total_popnum = 0
                for k in range(len(image_list)):
                # Define variables to store the best pair of images and their similarity score
                    best_pair = (None, None)
                    best_score = 0
                    
                    image2 = image_list[k]

                    # Try all possible combinations of rotations and flips for the two images
                    
                    for rotation2 in [0,90,180,270]:
                        
                        for horizontal_flip2 in [False, True]:
                            
                            for vertical_flip2 in [False, True]:
                                # Transform the two images based on the current flags
                                transformed_image1 = transform_image(test_image, 0, False, False)
                                transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                                # Compute the similarity score between the two transformed images
                                score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                # If the score is better than the previous best score, update the best pair
                                if score > best_score:
                                    best_pair = (transformed_image1, transformed_image2)
                                    best_score = score
                                    popnum = k

                    if best_score > total_best_score:

                        total_best_score = best_score
                        total_best_pair = best_pair
                        total_popnum = popnum



                test_image_rotate = transform_image(test_image,90,False,False)

                total_best_score_rotate = 0
                total_best_pair_rotate = (0,0)
                total_popnum_rotate = 0
                for k in range(len(image_list)):
                # Define variables to store the best pair of images and their similarity score
                    best_pair = (None, None)
                    best_score = 0
                    image2 = image_list[k]

                    # Try all possible combinations of rotations and flips for the two images
                    
                    for rotation2 in [0,90,180,270]:
                        
                        for horizontal_flip2 in [False, True]:
                            
                            for vertical_flip2 in [False, True]:
                                # Transform the two images based on the current flags
                                transformed_image1 = transform_image(test_image_rotate, 0, False, False)
                                transformed_image2 = transform_image(image2, rotation2, horizontal_flip2, vertical_flip2)

                                # Compute the similarity score between the two transformed images
                                score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                # If the score is better than the previous best score, update the best pair
                                if score > best_score:
                                    best_pair = (transformed_image1, transformed_image2)
                                    best_score = score
                                    popnum = k

                    if best_score > total_best_score:

                        total_best_score_rotate = best_score
                        total_best_pair_rotate = best_pair
                        total_popnum_rotate = popnum

                if total_best_score > total_best_score_rotate:
                    image_list.pop(total_popnum)
                    answersheet[minnum-1].append(total_best_pair[1])
                    for fill in range(minnum-1):
                        image1 = answersheet[fill][minnum-1]
                        popnum = 0
                        total_best_score = 0
                        total_best_pair = (0,0)
                        total_popnum = 0

                        for k in range(len(image_list)):
                        # Define variables to store the best pair of images and their similarity score
                            best_pair = (None, None)
                            best_score = 0
                            image2 = image_list[k]
                            # Try all possible combinations of rotations and flips for the two images
                            
                            
                            for horizontal_flip2 in [False, True]:
                                
                                for vertical_flip2 in [False, True]:
                                    # Transform the two images based on the current flags
                                    transformed_image1 = transform_image(image1, 0, False, False)
                                    transformed_image2 = transform_image(image2, 0, horizontal_flip2, vertical_flip2)

                                    # Compute the similarity score between the two transformed images
                                    score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                    # If the score is better than the previous best score, update the best pair
                                    if score > best_score:
                                        best_pair = (transformed_image1, transformed_image2)
                                        best_score = score
                                        popnum = k

                            if best_score > total_best_score:

                                total_best_score = best_score
                                total_best_pair = best_pair
                                total_popnum = popnum

                    
                        answersheet[fill].append(total_best_pair[1])
                        image_list.pop(total_popnum)
                else:
                    image_list.pop(total_popnum_rotate)
                    answersheet.append([0,0,0])
                    answersheet[minnum][minnum-1] = transform_image(total_best_pair_rotate[1],270,False,False)
                    for fill in range(minnum-1):
                        image1 = transform_image(answersheet[minnum][minnum-1-fill],0,True,False)
                        popnum = 0
                        total_best_score = 0
                        total_best_pair = (0,0)
                        total_popnum = 0

                        for k in range(len(image_list)):
                        # Define variables to store the best pair of images and their similarity score
                            best_pair = (None, None)
                            best_score = 0
                            image2 = image_list[k]
                            # Try all possible combinations of rotations and flips for the two images
                            
                            
                            for horizontal_flip2 in [False, True]:
                                
                                for vertical_flip2 in [False, True]:
                                    # Transform the two images based on the current flags
                                    transformed_image1 = transform_image(image1, 0, False, False)
                                    transformed_image2 = transform_image(image2, 0, horizontal_flip2, vertical_flip2)

                                    # Compute the similarity score between the two transformed images
                                    score = image_similarity(transformed_image1, transformed_image2, 'right', 'left')

                                    # If the score is better than the previous best score, update the best pair
                                    if score > best_score:
                                        best_pair = (transformed_image1, transformed_image2)
                                        best_score = score
                                        popnum = k

                            if best_score > total_best_score:

                                total_best_score = best_score
                                total_best_pair = best_pair
                                total_popnum = popnum

                        answersheet[minnum][minnum-1-fill-1] = transform_image(total_best_pair[1],0,True,False)
                        image_list.pop(total_popnum)


            merged_image = Image.new("RGB", (width * len(answersheet[0]), height * len(answersheet)))

            # Paste the best pair of images on the left and right side of the merged image
            for i in range(len(answersheet)):
                for j in range(len(answersheet[0])):
                    merged_image.paste(answersheet[i][j], (j*width,i*height))

            # Save the merged image
            merged_image.save(f"{opt.output_filename}.jpg")







        
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)