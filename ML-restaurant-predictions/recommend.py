from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map

##################################
# Phase 2: Unsupervised Learning #
##################################


def find_closest(location, centroids):
    """Return the centroid in centroids that is closest to location. If
    multiple centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [2.0, 3.0], [4.0, 3.0], [5.0, 5.0]])
    [2.0, 3.0]
    """
    centroid_distances = [(distance(location,centroid), centroid) for centroid in centroids]
    return min(centroid_distances)[1]


def group_by_first(pairs):
    """Return a list of pairs that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    >>> r1 = make_restaurant('X', [4, 3], [], 3, [
    ...         make_review('X', 4.5),
    ...      ]) # r1's location is [4,3]
    >>> r2 = make_restaurant('Y', [-2, -4], [], 4, [
    ...         make_review('Y', 3),
    ...         make_review('Y', 5),
    ...      ]) # r2's location is [-2, -4]
    >>> r3 = make_restaurant('Z', [-1, 2], [], 2, [
    ...         make_review('Z', 4)
    ...      ]) # r3's location is [-1, 2]
    >>> c1 = [4, 5]
    >>> c2 = [0, 0]
    >>> groups = group_by_centroid([r1, r2, r3], [c1, c2])
    >>> [[restaurant_name(r) for r in g] for g in groups]
    [['X'], ['Y', 'Z']] # r1 is closest to c1, r2 and r3 are closer to c2
    """
    clusters = []
    for restaurant in restaurants:
        closest_centroid = find_closest(restaurant_location(restaurant), centroids)
        centroid = str(closest_centroid)
        clusters.append([centroid, restaurant])
    
    return group_by_first(clusters)


def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster.
    >>> r1 = make_restaurant('X', [4, 3], [], 3, [
    ...         make_review('X', 4.5),
    ...      ]) # r1's location is [4,3]
    >>> r2 = make_restaurant('Y', [-3, 1], [], 4, [
    ...         make_review('Y', 3),
    ...         make_review('Y', 5),
    ...      ]) # r2's location is [-3, 1]
    >>> r3 = make_restaurant('Z', [-1, 2], [], 2, [
    ...         make_review('Z', 4)
    ...      ]) # r3's location is [-1, 2]
    >>> cluster = [r1, r2, r3]
    >>> find_centroid(cluster)
    [0.0, 2.0]
    """
    lats = []
    lons = []
    for restaurant in cluster:
        lat, lon = restaurant_location(restaurant)
        lats.append(lat)
        lons.append(lon)
    centroid_lat =  mean(lats)
    centroid_lon = mean(lons)
    return [centroid_lat, centroid_lon]


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    previous_centroids = []
    n = 0
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]
    while previous_centroids != centroids and n < max_updates:
        previous_centroids = centroids
        clusters = group_by_centroid(restaurants,centroids)
        centroids = [find_centroid(cluster) for cluster in clusters]
        n += 1
    return centroids

def find_predictor(user, restaurants, feature_fn):
    """Return a score predictor (a function from restaurants to scores),
    for a user by performing least-squares linear regression using feature_fn
    on the items in restaurants. Also, return the R^2 value of this model.

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    reviews_by_user = {review_restaurant_name(review): review_score(review)
                       for review in user_reviews(user).values()}

    xs = [feature_fn(r) for r in restaurants]
    ys = [reviews_by_user[restaurant_name(r)] for r in restaurants]

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    s_xx = sum([(xi - mean_x) ** 2 for xi in xs])
    s_yy = sum([(yi - mean_y) ** 2 for yi in ys])
    s_xy = sum([(xi - mean_x) * (yi - mean_y) for xi, yi in zip(xs, ys)])

    b = s_xy / s_xx
    a = mean_y - b * mean_x 
    r_squared = (s_xy ** 2) / (s_xx * s_yy) 

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting scores by the user; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)
    predictors = []
    for function in feature_fns:
        predictor, r_squared = find_predictor(user,reviewed, function)
        predictors.append((predictor, r_squared))
    highest_r_squared = max(predictors, key = lambda x: x[1])[0]

    return highest_r_squared


def rate_all(user, restaurants, feature_fns):
    """Return the predicted scores of restaurants by user using the best
    predictor based a function from feature_fns.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)
    rest_dict ={}
    reviewed_names = [restaurant_name(r) for r in reviewed]
    for restaurant in restaurants:
        rest_name = restaurant_name(restaurant)
        if rest_name in reviewed_names:
            user_scored = user_score(user, rest_name)
            if isinstance(user_scored, float):
                rest_dict[rest_name] = int(user_scored) if user_scored.is_integer() else user_scored
            else:
                rest_dict[rest_name] = user_scored
        else:
            predicted = predictor(restaurant)
            rest_dict[rest_name] = predicted
    return rest_dict


def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    return [restaurant for restaurant in restaurants if query in restaurant_categories(restaurant)]


def feature_set():
    """Return a sequence of feature functions."""
    return [restaurant_mean_score,
            restaurant_price,
            restaurant_num_scores,
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]

@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict scores for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_score(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)
