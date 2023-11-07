def split_text(text, max_length=2000):
    """
    Split the text into pieces of at most max_length characters. 
    The function prioritizes splitting at paragraph breaks, then periods, and finally spaces.
    It tries to keep the pieces about equally sized.
    
    Args:
    - text (str): The input text.
    - max_length (int): The maximum length of each piece. Default is 2000.

    Returns:
    - List[str]: List of text pieces.
    """
    
    # Split by paragraph first
    paragraphs = text.split('\n')
    pieces = []
    current_piece = ""

    for paragraph in paragraphs:
        # If the current piece + the new paragraph is too long
        if len(current_piece + paragraph) > max_length:
            # If the current piece is not empty, add it to the pieces
            if current_piece:
                pieces.append(current_piece)
                current_piece = ""
            
            # If the paragraph itself is longer than max_length, split it further
            while len(paragraph) > max_length:
                # Find the last period within max_length
                split_point = paragraph.rfind('.', 0, max_length)
                
                # If there's no period, find the last space within max_length
                if split_point == -1:
                    split_point = paragraph.rfind(' ', 0, max_length)
                
                # If there's neither a period nor a space, just split at max_length
                if split_point == -1:
                    split_point = max_length
                
                # Add the split part to the pieces and remove it from the paragraph
                pieces.append(paragraph[:split_point + 1].strip())
                paragraph = paragraph[split_point + 1:].strip()
            
            # Add the remainder of the paragraph to the current piece
            current_piece = paragraph
        else:
            # If the paragraph can be added to the current piece without exceeding max_length
            if current_piece:
                current_piece += "\n" + paragraph
            else:
                current_piece = paragraph

    # If there's any remaining text in the current piece, add it to the pieces
    if current_piece:
        pieces.append(current_piece)

    return pieces
    