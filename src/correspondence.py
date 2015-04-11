from scipy.spatial import cKDTree as KDTree

ID = 0
def new_data(feature):
   global ID
   ID += 1
   return {
      'id': ID
   }

# corresponds two sets of features
# old_data = data associated with the old features
# outputs:
#    features: features that carried over from old to new
#    feature_data: the respective data for those features (from old_data)
#    missing_features: features that were in new_features and not in old_features
def correspond(old_data, old_features, new_features, threshold=100, d=2):
   # trim to dimensional coordinates
   old_features_d = [feature[:d] for feature in old_features]
   new_features_d = [feature[:d] for feature in new_features]

   if len(old_features_d) == 0:
      nearest_new_feature_indexes = []
      distances = []
   else:
      # find the nearest new feature for each old feature
      distances, nearest_new_feature_indexes = KDTree(new_features_d).query(old_features_d, distance_upper_bound=threshold)

   # to keep track of any features that have disappeared from old_features
   # and are not nearby in new_features
   missing_feature_indexes = set()

   # assignments of new feature locations to old feature locations
   nearest_neighbours = {}

   size = len(new_features) # how many new features

   # iterate through each nearest neighbour pair
   for old_i, nearest_new_i in enumerate(nearest_new_feature_indexes):
      distance = distances[old_i]

      if distance < threshold and nearest_new_i < size:
         # the old feature was indeed assigned a nearest new feature

         if nearest_new_i in nearest_neighbours:
            # the nearest new feature has already been matched to an old one

            corresponding_old_i, prev_distance = nearest_neighbours[nearest_new_i]
            if prev_distance > distance:
               # this one is better

               # assign the old one as missing (we've ditched its previous assignment)
               missing_feature_indexes.add(corresponding_old_i)

               # this candidate is closer, so it's the new assignment
               nearest_neighbours[nearest_new_i] = (old_i, distance)
         else:
            # no assignment has been made to this new feature, so we assign this one
            nearest_neighbours[nearest_new_i] = (old_i, distance)
      else:
         # there is no nearby new feature, this old feature has gone missing
         missing_feature_indexes.add(old_i)

   # the new feature data
   feature_data = []
   features = []

   # collate assigned features
   for new_i, new_feature in enumerate(new_features):
      if new_i not in nearest_neighbours:
         # not assigned to an old feature, it must be new
         features.append(new_feature)
         new_data_point = new_data(new_feature)
         new_data_point.update({
            'feature': new_feature,
            'distance': -1
         })
         feature_data.append(new_data_point)
      else:
         # assigned to an old one, update the data
         old_i, distance = nearest_neighbours[new_i]
         old_data_point = old_data[old_i]
         new_data_point = {}
         new_data_point.update(old_data_point)
         new_data_point.update({
            'feature': new_feature,
            'distance': distance
         })
         features.append(new_feature)
         feature_data.append(new_data_point)

   # data for features that have gone missing
   missing_features = []
   for missing_i in missing_feature_indexes:
      missing_data_point = {}
      missing_data_point.update(old_data[missing_i])
      missing_features.append(missing_data_point)

   return {
      'features': features,
      'feature_data': feature_data,
      'missing_features': missing_features,
   }
