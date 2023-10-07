def get_iou(bbox_a, bbox_b):
    x1_intersection = max(bbox_a[0], bbox_b[0])
    y1_intersection = max(bbox_a[1], bbox_b[1])
    x2_intersection = min(bbox_a[2], bbox_b[2])
    y2_intersection = min(bbox_a[3], bbox_b[3])
    intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(0, y2_intersection - y1_intersection + 1)
    bbox_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    bbox_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    iou = intersection_area / float(bbox_a_area + bbox_b_area - intersection_area)
    return iou


def distance(a, b):
    """Calculates the Levenshtein distance between a and b."""
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n, m)) space
        a, b = b, a
        n, m = m, n

    current_row = range(n + 1)  # Keep current and previous row, not entire matrix
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]


class LPfilter:
    def __init__(self, history_length=200, max_text_distance=3, min_iou=0.5):
        self.__min_iou = min_iou
        self.__max_text_distance = max_text_distance
        self.__history_length = history_length
        self.__lp_list = []

        # self.__lp_list.append(
        #     {
        #         'bbox': (1, 2, 3, 4),
        #         'first_frame_seen': 2,
        #         'last_frame_seen': 2,
        #         'text': 'asdasdsa',
        #     }
        # )

    def clear_history(self, frame_number):
        lp_list_new = []
        for lp in self.__lp_list:
            if frame_number - lp['last_frame_seen'] < self.__history_length:
                lp_list_new.append(lp)
        self.__lp_list = lp_list_new

    def get_lp_first_seen_frame(self, bbox, text, frame_number):
        lp_list_new = []
        first_frame_seen = frame_number
        was_added = False
        if len(self.__lp_list) > 0:
            for lp in self.__lp_list:
                # print('iou:', get_iou(lp['bbox'], bbox))
                # print('dist:', distance(lp['text'], text))

                # if it was seen lp we refresh data
                if (get_iou(lp['bbox'], bbox) > self.__min_iou) or (distance(lp['text'], text) < self.__max_text_distance):
                    lp_list_new.append({
                        'bbox': bbox,
                        'first_frame_seen': lp['first_frame_seen'],
                        'last_frame_seen': frame_number,
                        'text': text,
                    })
                    first_frame_seen = lp['first_frame_seen']
                    was_added = True
                # keep not seen lp in memory
                else:
                    lp_list_new.append(lp)

            # if a new one lp we add it
            if not was_added:
                lp_list_new.append({
                    'bbox': bbox,
                    'first_frame_seen': frame_number,
                    'last_frame_seen': frame_number,
                    'text': text,
                })

        else:
            lp_list_new.append({
                'bbox': bbox,
                'first_frame_seen': frame_number,
                'last_frame_seen': frame_number,
                'text': text,
            })

        self.__lp_list = lp_list_new
        return first_frame_seen
