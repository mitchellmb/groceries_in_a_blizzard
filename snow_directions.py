import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import folium
import openrouteservice as ors
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, shape 
from sklearn.cluster import KMeans
pd.options.mode.chained_assignment = None 
pd.set_option('display.max_columns', None)

#%%
#Data from Open Data Minneapolis (https://opendata.minneapolismn.gov/). 
#Snow emergency data was cleaned and combined from 2015-2023.
#Police incidents data between 2013-2023 was cleaned, combined, and filtered for car break-in and theft.
snow_df = pd.read_csv('snow_emergency_data_processed.csv', index_col=0)
crime_df = pd.read_csv('car_crimes_data_processed.csv', index_col=0)

#List of grocery store locations in Minneapolis, MN and their names
grocery_df = pd.read_csv('grocery_data_processed.csv', index_col=0)

#Openroute Service API
ORS_API = os.getenv('ORS_API')
client = ors.Client(key=ORS_API)
#%%

def get_long_lat(addr):
    '''Geocode a string address with openrouteservice API'''
    geocode = client.pelias_search(text = addr,
                                   #(long,lat) centered in Minneapolis, MN
                                   focus_point = [-93.275, 44.97], 
                                   validate = True)
    coords = geocode['features'][0]['geometry']['coordinates']
    time.sleep(4) #API request limitations
    return coords

def store_match(store, row):
    '''Check if the first name of the strings store and row match.'''
    store = store.split(' ')[0].lower()
    row = row.split(' ')[0].lower()
    
    if row == store:
        return True
    return False

latitude = 69.2 #miles/degree
longitude = 54.6 #miles/degree
def circle(x, y, radius, center_x, center_y):
    '''Check if a location (x, y) pair is within a given radius of (center_x, center_y).
    Used to check if a store's (x, y) is within the radius from the home's (center_x, center_y).'''
    #Convert longitude/latitude degrees to miles
    x_dist = np.abs(x-center_x)*longitude
    y_dist = np.abs(y-center_y)*latitude
    
    #Calcualte the square of distance store-home
    tot_dist = (x_dist)**2.+(y_dist)**2.
    
    #Check if it is inside the radius, compares both distances squared
    if tot_dist < radius**2.:
        return True
    return False


def poly_make(row, polygon_scale):
    '''Create a close-looped polygon from snow cluster center event data (row). This returns a rectangle 
    that is used as an input to the route generation. The rectangle is multipled by polygon_scale to 
    increase or decrease the output rectangle's size.'''
    
    x_min = row['Long min']
    x_max = row['Long max']
    y_min = row['Lat min']
    y_max = row['Lat max']
    
    x_len = np.abs(x_max-x_min)
    y_len = np.abs(y_max-y_min)
    
    x_mid = x_min+(x_len/2.)
    y_mid = y_min+(y_len/2.)
    
    # Reduce size if metric is small
    metric = row['Metric norm']
    
    #Scale the polygon by metric and by the overall scale factor, polygon_scale
    x_len_scaled = x_len*metric*polygon_scale
    y_len_scaled = y_len*metric*polygon_scale
    
    x_min_scaled = x_mid-x_len_scaled
    x_max_scaled = x_mid+x_len_scaled
    y_min_scaled = y_mid-y_len_scaled
    y_max_scaled = y_mid+y_len_scaled
    
    #Set up JSON Format
    poly = [[[x_min_scaled, y_min_scaled], [x_max_scaled, y_min_scaled],
            [x_max_scaled, y_max_scaled], [x_min_scaled, y_max_scaled],
            [x_min_scaled, y_min_scaled]]] #5th one is a repeat of 1st to 'close' the polygon
            
    return poly

def center(min_, max_):
    '''Find the center point of a line, used for finding the center of polygons.'''
    return (min_ + max_)/2. 


def cluster_rank(severity, clust_x, clust_y, store_x, store_y):
    '''Calculate the rank of a store relative to a given snow or crime cluster center. Severity is a 
    multiplicative factor proportional to the number and type of events in the given cluster. 
    clust_x & clust_y are the center locations of a cluster. store_x and store_y are the locations of 
    a store.'''
    
    #Distance in miles
    dist = np.sqrt(((clust_x - store_x)*longitude)**2. 
                 + ((clust_y - store_y)*latitude)**2.)
    
    #Metric of a store is proportional to the severity & distance to a nearby cluster event
    rank_metric = severity*dist
    
    return rank_metric

#%%

def route(start, end):
    '''Generate a route with openrouteservice API. This function does not avoid any snow or crime 
    clusters.'''
    
    request_params = {'coordinates': [start, end],
                     'format_out': 'geojson',
                     'profile': 'driving-car',
                     'preference': 'shortest',
                     'instructions': True}
    route_directions = client.directions(**request_params)

    return route_directions

def route_plus_avoid(start, end, avoided_polys):
    '''Generate a route with openrouteservice API. This function also avoids
    polygons.'''
    
    request_params = {'coordinates': [start, end],
                     'format_out': 'geojson',
                     'profile': 'driving-car',
                     'preference': 'shortest',
                     'instructions': True,
                     'options': {'avoid_polygons': avoided_polys}}
    route_directions = client.directions(**request_params)

    return route_directions

def check_route_intersect(polygon, route_coords):
    '''Check if a given polygon shape is intersected by route_coords directions.'''
    
    poly = Polygon(polygon)
    
    for i in range(len(route_coords)-1):
        line = LineString([route_coords[i], route_coords[i+1]])
        
        #Check if any line segment intersects the given polygon, return True if so
        if line.intersects(poly):
            return True
        
    return False    

#%%

class SnowDirections():
    
    '''
    A class to find and plot directions from a location in Minneapolis, MN to a given grocery store.
    
    ...
    
    Attributes
    ----------
    snow: dataframe
        Dataframe data of vehicle snow events - location (latitude/longitude), type (tag or tow), date
    groc: dataframe
        Dataframe data of grocery stores in Minneapolis, MN - location (latitude/longitude), name
    crim: dataframe
        Dataframe data of vehicle crime events - location (latitude/longitude), type (theft or break-in),
        date
        
    store: string
        Preferred grocery store input name
    store_matches: list
        Stores within radius that match the input store's name
    closest_store_dist: float
        Distance to closest store. Default 100.
    closest_store: string
        Name of closest store matching input store's name
    
    home_addr: string
        Starting location address
    radius: integer
        Maximum distance to search around home_addr
    home: list
        List of floats of longitude & latitude values from home_addr geolocation
    home_point: Point
        Geometry Point class of home
    in_minneapolis: bool
        Bool for if the home is inside Minneapolis, MN city limits
        
    polygon_scale: float
        Multiplicative factor of snow event clusters used in route generation
    snow_poly: dataframe
        Minimum and maximum of longitude and latitude and severity values for each snow_centers cluster,
        where severity is the average of: (event type) * (number of events). Event types are: tag = 1,
        tow = 2, break-in = 3, theft = 4.
    crim_poly: dataframe
        Minimum and maximum of longitude and latitude and severity values for each crim_centers cluster,
        where severity is the average of: (event type) * (number of events). Event types are: tag = 1,
        tow = 2, break-in = 3, theft = 4.
    poly_intersected: list
        List of polygons that intersect the current route
    poly_intersected_tot: list
        List of polygons that intersected current and previous route attempts        
    polys_to_avoid: list
        Snow cluster polygons that should be avoided when generating a route. Does not include polygons
        that contain the home address.
    polys_cover_home: list
        Snow cluster polygons containing the home address inside of them.

    n_clusters: integer
        Number of clusters for k-means
    snow_centers: dataframe
        K-means clustered values of snow dataframe input data
    crim_centers: dataframe
        K-means clustered values of crime dataframe input data    
    
    checked: bool
        Whether a route has been checked to avoid snow even cluster polygons or not
    route_directions: dictionary
        JSON formatted dictionary of directions of current route
    route_rescale_reset: dictionary
        JSON formatted dictionary of original first pass route without avoiding snow cluster polygons    
    steps: dictionary
        Route steps
    instructions: list
        Route steps in a list
        
    Methods
    -------
    geolocate_address():
        Geolocates the home address and checks if it is in Minneapolis, MN
    store_pref():
        Finds grocery stores that match the preferred input grocery store and the closest store.
    data_trim():
        Trims snow, groc, crim dataframes to within the input radius limit
    cluster():
        K-means clustering of snow and crime events within the radius
    snow_clust_df():
        Determines the latitude and longitude limits and severity of each snow cluster
    crim_clust_df():
        Determines the latitude and longitude limits and severity of each crime cluster
    rank_stores():
        Ranks grocery stores within radius by how far they are away from the home address, how many
        snow clusters & how severe the snow clusters are nearby, and how many crime clusters & how 
        severe the crime clusters are nearby. Output as new column in self.groc dataframe.
    poly_scale(rescale=False, rescaler=0.9):
        Takes snow cluster events and returns polygons to input into API routing, with rescale
        defining if the polygons are scaled-down by rescaler
    find_route():
        Determines the initial route from home to store, without avoiding snow cluster polygons
    check_route():
        Recursively checks the initial route against snow cluster polygons to find an
        optimized route avoiding as many polygons as possible.
    instructions():
        Returns the instructions for the current route
    plotting(crime_clust=True, snow_clust=True, poly_plot=False):
        Folium plot of current route, home address, preferred grocery store(s), other stores,
        snow clusters, crime clusters, and optionally the snow cluster polygon shapes used
        in generating the route
    '''
    
    def __init__(self, snow, groc, crim, store, home_addr, radius, polygon_scale): 
        '''Constructs necessary attributes for the SnowDirections object'''
        
        self.snow = snow
        self.groc = groc
        self.crim = crim
        self.store = store
        self.addr = home_addr
        self.radius = radius
        self.polygon_scale = polygon_scale
        
        #Setup for route checking
        self.checked = False #if a route has been checked
        self.poly_intersected = [] #list of polygons the routes have intersected, resets on rescaling
        self.poly_intersected_tot = [] #no reset on rescale
        
        #clusters per square mile
        sq_mi = np.pi*(radius)**2. 
        self.n_clusters = round(10*sq_mi)
    
    def geolocate_address(self):
        '''Finds the geolocation in latitude/longitude of the home address string'''
        
        self.home = get_long_lat(self.addr)
        #geometry for checking route polygon intersection in check_route()
        self.home_point = Point(self.home[0], self.home[1]) 
        
        #check if the address is in Minneapolis, MN
        long_lims = [-93.38, -93.16]
        lat_lims = [44.86, 45.08]
        
        if self.home[0] >= long_lims[0] and self.home[0] <= long_lims[1]:
            if self.home[1] >= lat_lims[0] and self.home[1] <= lat_lims[1]:
                self.in_minneapolis = True
        else:
            self.in_minneapolis = False
        
    def store_pref(self):
        '''Finds the preferred grocery store from the list of grocery stores in Minneapolis, MN
        The closest version of that store is named closest_store'''
        
        self.groc['store match'] = self.groc['Name'].apply(lambda x: store_match(self.store, x))
        self.store_matches = [store for index, store in self.groc.iterrows() 
                              if store['store match'] == True]
        
        if self.store_matches:
            #default values for closest store
            self.closest_store_dist = 100 
            self.closest_store = self.store_matches[0]

            for store in self.store_matches:
                #Find distance from start to store_match, converts to miles
                lat_diff = np.abs(self.home[1] - store['Latitude'])*latitude 
                long_diff = np.abs(self.home[0] - store['Longitude'])*longitude 
                dist_to_store = np.sqrt(lat_diff**2.+long_diff**2.)
                
                #replace default values for closest store
                if dist_to_store < self.closest_store_dist:
                    self.closest_store = store
                    self.closest_store_dist = dist_to_store
            
    def data_trim(self):
        '''Removes input data outside of the radius around the home address'''
        
        #set a new column in self.groc to flag if the grocery store is in the radius
        self.groc['in_radius'] = self.groc.apply(
            lambda x: circle(x['Longitude'], x['Latitude'], self.radius, self.home[0], self.home[1]), 
            axis=1)
        
        #repeat with snow and crime data, but expand the radius to allow routes to go
        #slightly outside of the distance to the grocery store
        route_radius = self.radius*1.5      
        
        self.snow['in_radius'] = self.snow.apply(
            lambda x: circle(x['Longitude'], x['Latitude'], route_radius, self.home[0], self.home[1]),
            axis=1)
        
        self.crim['in_radius'] = self.crim.apply(
            lambda x: circle(x['Longitude'], x['Latitude'], route_radius, self.home[0], self.home[1]),
            axis=1)
        
        #drop rows not in radius
        self.snow = self.snow[self.snow['in_radius'] == True]
        self.groc = self.groc[self.groc['in_radius'] == True]
        self.crim = self.crim[self.crim['in_radius'] == True]
        
    def cluster(self):
        '''K-means cluster snow-related events (tag & tows) from self.snow and crime-related
        events (break-in & theft) from self.crim'''
        
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=5, random_state=42)
        
        snow_cluster = kmeans.fit(self.snow[['Longitude', 'Latitude']])
        self.snow_centers = pd.DataFrame(snow_cluster.cluster_centers_, 
                                         columns=['Long cent', 'Lat cent'])
        self.snow['snow_cluster'] = snow_cluster.predict(self.snow[['Longitude', 'Latitude']])
        
        crim_cluster = kmeans.fit(self.crim[['Longitude', 'Latitude']])
        self.crim_centers = pd.DataFrame(crim_cluster.cluster_centers_, 
                                         columns=['Long cent', 'Lat cent'])
        self.crim['crim_cluster'] = crim_cluster.predict(self.crim[['Longitude', 'Latitude']])
        
    def snow_clust_df(self):
        '''Using the snow clusters, create a dataframe that stores information on each cluster.
        This includes: latitude & longitude minimum and maximum, count, severity, and a metric. The 
        metric is proportional to the number of events within the cluster multiplied by the average
        event type severity (tag = 1, tow = 2) so that tagging events are less 'costly' than towing 
        events in the overall metric of the cluster.'''
        
        #in a cluster: 
        #metric = [(number of events) * (mean of severity)]
            #gives events/mean normalized
            #makes metric large if:
                #counts are large or severity is large
        
        x_min=[]
        x_max=[]
        y_min=[]
        y_max=[]
        severity=[]
        count=[]
        metric=[]
        cluster=[]
        for clust in self.snow['snow_cluster'].unique():
            filtered = self.snow[self.snow['snow_cluster'] == clust]
            
            x_min.append(filtered['Longitude'].min())
            x_max.append(filtered['Longitude'].max())
            
            y_min.append(filtered['Latitude'].min())
            y_max.append(filtered['Latitude'].max())
            
            severity.append(filtered['Severity'].mean())
            count.append(len(filtered))
            metric.append(len(filtered)*filtered['Severity'].mean())
            
            cluster.append(clust)
            
        metric = np.array(metric)
        metric_norm = metric/np.max(metric) #normalizes for scaling the polygons later    
        
        x_center = center(np.array(x_min), np.array(x_max))
        y_center = center(np.array(y_min), np.array(y_max))
        
        s = {'snow cluster':cluster, 
             'Long min': x_min, 'Long max': x_max, 'Long center': x_center,
             'Lat min': y_min, 'Lat max': y_max, 'Lat center': y_center,
             'Severity avg':severity, 'Number of events':count, 
             'Metric norm':metric_norm}
        
        self.snow_poly = pd.DataFrame(data=s)     
        self.snow_poly = self.snow_poly.set_index('snow cluster').sort_index().join(self.snow_centers)
        
    def crim_clust_df(self): 
        '''See snow_clust_df. Severity is now break-ins = 3 and theft = 4'''

        x_min=[]
        x_max=[]
        y_min=[]
        y_max=[]
        severity=[]
        count=[]
        metric=[]
        cluster=[]
        for clust in self.crim['crim_cluster'].unique():
            filtered = self.crim[self.crim['crim_cluster'] == clust]
            
            x_min.append(filtered['Longitude'].min())
            x_max.append(filtered['Longitude'].max())
            
            y_min.append(filtered['Latitude'].min())
            y_max.append(filtered['Latitude'].max())
            
            severity.append(filtered['Severity'].mean())
            count.append(len(filtered))
            metric.append(len(filtered)*filtered['Severity'].mean())
            
            cluster.append(clust)
        
        metric = np.array(metric)
        metric_norm = metric/np.max(metric)
        
        x_center = center(np.array(x_min), np.array(x_max))
        y_center = center(np.array(y_min), np.array(y_max))
        
        c = {'crim cluster':cluster, 
             'Long min': x_min, 'Long max': x_max, 'Long center': x_center,
             'Lat min': y_min, 'Lat max': y_max, 'Lat center': y_center,
             'Severity avg':severity, 'Number of events':count, 
             'Metric norm':metric_norm}
        
        self.crim_poly = pd.DataFrame(data=c)
        self.crim_poly = self.crim_poly.set_index('crim cluster').sort_index().join(self.crim_centers)
        
    
    
    def rank_stores(self):
        '''Ranks grocery stores within radius by how far they are away from the home address, how many
        snow clusters & how severe the snow clusters are nearby, and how many crime clusters & how 
        severe the crime clusters are nearby. Output as new column in self.groc dataframe.'''
        
        #Ranking metric: prioritize close & low-severity stores with a low overall rank_metric
        
        store_longs = np.array(self.groc['Longitude'])
        store_lats = np.array(self.groc['Latitude'])
        
        long_diff = np.array([(self.home[0]-long)*longitude for long in store_longs])
        lat_diff = np.array([(self.home[1]-lat)*latitude for lat in store_lats])
        
        #Distance to a store from home address
        dist_to_store = np.sqrt(long_diff**2. + lat_diff**2.)
        
        crime_rankings=[]
        for i in range(0, len(store_longs)):
            rank = self.crim_poly.apply(lambda row: cluster_rank(row['Severity avg'], 
                                                                 row['Long center'], row['Lat center'],
                                                                 store_longs[i], store_lats[i]),
                                        axis = 1)
            rank_tot = np.sum(rank)
            
            #Weight this store's rankings additionally by distance from home address, 
            #normalized to total crime clusters       
            rank_tot_weighted = rank_tot*dist_to_store[i]/len(self.crim_poly)
            
            crime_rankings.append(rank_tot_weighted)
            
        snow_rankings=[]
        for i in range(0, len(store_longs)):
            rank = self.snow_poly.apply(lambda row: cluster_rank(row['Severity avg'], 
                                                                 row['Long center'], row['Lat center'],
                                                                 store_longs[i], store_lats[i]),
                                        axis = 1)
            
            rank_tot = np.sum(rank)
            rank_tot_weighted = rank_tot*dist_to_store[i]/len(self.snow_poly)
            
            snow_rankings.append(rank_tot_weighted)
        
        #Store the ranks in the grocery dataframe
        self.groc['Crime rank'] = crime_rankings
        self.groc['Snow rank'] = snow_rankings
        self.groc['Total rank'] = self.groc['Crime rank'] + self.groc['Snow rank']
        
        #Reorder list by total rank
        self.groc = self.groc.sort_values(by='Total rank', ascending=True).reset_index()
                
    
    def poly_scale(self, rescale=False, rescaler=0.9):
        '''Takes snow cluster events and returns polygons to input into API routing, with rescale
        defining if the polygons are scaled-down by rescaler'''
              
        if rescale:
            self.route_directions = self.route_rescale_reset
            self.poly_intersected = []
            
            self.polygon_scale = self.polygon_scale*rescaler
            snow_poly_list = [poly_make(row[-1], self.polygon_scale)[0] 
                              for row in self.snow_poly.iterrows()]
            
            self.polys_all = {'coordinates':snow_poly_list, 'type':'Polygon'}
            
        else:
            snow_poly_list = [poly_make(row[-1], self.polygon_scale)[0] 
                              for row in self.snow_poly.iterrows()]

            self.polys_all = {'coordinates':snow_poly_list, 'type':'Polygon'}

    def find_route(self):
        '''Generate an initial route without avoiding polygons'''
        
        self.start = (self.home[0], self.home[1])
        
        self.end = (self.closest_store['Longitude'], self.closest_store['Latitude'])
        self.route_directions = route(self.start, self.end)
    
        #if rescale is called later on, we have to reset the route to the initial guess
        self.route_rescale_reset = self.route_directions 
         
    
    
    def check_route(self):
        '''Recursively checks the initial route against snow cluster polygons to find an
        optimized route avoiding as many polygons as possible.'''
        
        #Flag for plotting
        self.checked = True
        
        #Polygon setup:
        #(1) Get directions
        route_dir = self.route_directions['features'][0]['geometry']['coordinates']
        
        #(2) List of polygons to avoid, and those that cover home
        self.polys_to_avoid = [i for i in self.polys_all['coordinates'] 
                               if not Polygon(i).contains(self.home_point)]
        
        self.polys_cover_home = [i for i in self.polys_all['coordinates'] 
                                if Polygon(i).contains(self.home_point)]
        
        #(3) Polygons that intersect route
        polys_inter = [i for i in self.polys_to_avoid if check_route_intersect(i, route_dir)]          
        
        #(4) Append the polygons that intersect route (if they aren't already in the running list)
        if polys_inter:
            for i in polys_inter:
                if i not in self.poly_intersected:
                    self.poly_intersected.append(i)
                    self.poly_intersected_tot.append(i)
        
        
        #CHECKS: if fail, recursively re-scale -> re-run
        
        #CHECK 1: How many polys cover home address? (max 1 allowing)
        if (len(self.polys_all['coordinates']) - len(self.polys_to_avoid)) > 1:
            self.poly_scale(rescale=True) #remake self.polys_all
            self.check_route() 
        
        #CHECK 2: Max polygons 4 into API    
        if len(self.poly_intersected) > 4:
            self.poly_scale(rescale=True)
            self.check_route()
            
        #Make a route
        try:
            #ensure there are polygons to avoid at this scale!
            if self.poly_intersected: 
                self.route_directions = route_plus_avoid(self.start, self.end,
                                        {'coordinates':self.poly_intersected, 'type':'Polygon'})
            
            #CHECK 3: Does new route overlap with a different set of polygons?
            route_dir = self.route_directions['features'][0]['geometry']['coordinates']
            polys_inter_2 = [i for i in self.polys_to_avoid if check_route_intersect(i, route_dir)]
            
            #If new polygons overlap, re-run the check_route, else it exits
            if polys_inter_2:
                for i in polys_inter_2:
                    self.poly_intersected.append(i)
                    self.poly_intersected_tot.append(i)
                time.sleep(4)
                self.check_route()        
            
        except Exception:
            #A route was not possible with this polygon scale, re-scale -> re-run
            self.poly_scale(rescale=True)
            time.sleep(4)
            self.check_route()
                        
    
    def instructions(self):
        '''Finds the list of instructions for the current route'''
        self.steps = self.route_directions['features'][0]['properties']['segments'][0]['steps']
        self.instructions = [i['instruction'] for i in self.steps]
        
        inst_list = []
        for index, row in enumerate(self.instructions):
            inst_list.append('Step '+ str(index+1) + ':' + str(row))
            
        return inst_list
    
    
    def plotting(self, crime_clust=True, snow_clust=True, poly_plot=False):
        '''Creates a folium map of the Minneapolis, MN region centerd on the home address. 
        Plot includes home, preferred grocery store, other grocery stores, snow clusters,
        crime clusters, and optionally the snow clusters converted to polygons that were 
        avoided in route generation'''
        
        #Generate map around home address
        input_map = folium.Map(location=[self.home[1], self.home[0]], 
                               tiles='cartodbpositron', zoom_start = 13-(self.radius/2))
        
        #Home address
        folium.Marker(location=[self.home[1], self.home[0]], 
                      popup=f'<i>{self.addr}</i>',
                      tooltip='Home',
                      icon=folium.Icon(icon='home', color='green', prefix='fa'),
                      zoom_on_click=True).add_to(input_map)
        
        #Store preference
        for index, store in self.groc.iterrows():
            if store['store match'] == True:
                folium.Marker(location=[store['Latitude'], store['Longitude']],
                              popup=f'<i>{self.store}</i>',
                              tooltip=f'Preferred grocery: {self.store}',
                              icon=folium.Icon(icon='shopping-cart', color='beige', prefix='fa'),
                              opacity=1,
                              zoom_on_click=True).add_to(input_map)
        #Top 10 ranked stores 
            elif index in range(0,10):
                other_store = store['Name']
                folium.Marker(location=[store['Latitude'], store['Longitude']],
                              popup=f'<i>{other_store}</i>',
                              tooltip=f'Alternative: {other_store}',
                              icon=folium.Icon(icon='shopping-cart', color='black', prefix='fa'),
                              opacity=0.8,
                              zoom_on_click=True).add_to(input_map)
        #All other stores
            else:
                other_store = store['Name']
                folium.Marker(location=[store['Latitude'], store['Longitude']],
                              popup=f'<i>{other_store}</i>',
                              tooltip=f'{other_store}',
                              icon=folium.Icon(icon='store', color='lightgray', prefix='fa'),
                              opacity=0.6, z_index=0, 
                              zoom_on_click=True).add_to(input_map)
        
        #Snow clusters
        if snow_clust == True:
            for index, clust in self.snow_poly.iterrows():
                folium.CircleMarker(location=[clust['Lat center'], clust['Long center']], 
                                    tooltip='snow event cluster center',
                                    color='lightblue', fill_color='lightblue', fill_opacity=1,
                                    radius=clust['Metric norm']*20, weight=1).add_to(input_map)
            
        #Crime clusters
        if crime_clust == True:
            for index, clust in self.crim_poly.iterrows():
                folium.CircleMarker(location=[clust['Lat center'], clust['Long center']], 
                                    tooltip='crime event cluster center', 
                                    color='darkblue', fill_color='darkblue', fill_opacity=1,
                                    radius=clust['Metric norm']*15, weight=1).add_to(input_map)
            
        #Route
        folium.features.GeoJson(data=self.route_directions,
                                name='Snow incident avoid map',
                                overlay=True).add_to(input_map)
        
        #Polygons - conditional plotting of all polygons surrounding clusters in the area
        if poly_plot == True:
            
            polys = [poly for poly in self.polys_all['coordinates']]
            
            for i in polys:          
                
                temp_poly = {'coordinates': [i], 'type': 'Polygon'}
                temp_poly_df = gpd.GeoDataFrame({'geometry': [shape(temp_poly)]}).set_crs(epsg=4326)
                folium.GeoJson(data=temp_poly_df,
                               style_function=lambda x: {'color': '#8eaebf', 
                                                         'fillColor': '#8eaebf'}).add_to(input_map)
        
        #Polygons - conditional plotting that colors polygons avoided when gernerating the route
        if (self.checked == True) and (poly_plot == True):
            
            #Polygons that covered the home address in green
            for i in self.polys_cover_home:               
                temp_poly = {'coordinates': [i], 'type': 'Polygon'}
                temp_poly_df = gpd.GeoDataFrame({'geometry': [shape(temp_poly)]}).set_crs(epsg=4326)
                folium.GeoJson(data=temp_poly_df, #opacity=0.7,
                               style_function=lambda x: {'color': 'green', 
                                                         'fillColor': 'green'}).add_to(input_map)
            
            #Polygons that were avoided in bright blue
            for i in self.poly_intersected_tot:
                temp_poly = {'coordinates': [i], 'type': 'Polygon'}
                temp_poly_df = gpd.GeoDataFrame({'geometry': [shape(temp_poly)]}).set_crs(epsg=4326)
                folium.GeoJson(data=temp_poly_df,
                               style_function=lambda x: {'color': '#39aae6', 
                                                         'fillColor': '#39aae6'}).add_to(input_map)
            
        
        return input_map  
    
#%%

def directions(grocery_name, home_address, radius=2, polygon_scale=1, check_route=True):
    '''
    Generate a route in Minneapolis, MN to a grocery store, avoiding as many historical
    snow cluster events as possible.

    Parameters
    ----------
    grocery_name : string
        Preferred grocery store to route to.
    home_address : string
        Starting location.
    radius : integer, optional
        Maximum distance to route to surrounding home_address (in miles). The default is 2.
    polygon_scale : float, optional
        Starting point of routing polygons. These polygons are located at snow event 
        cluster centers. The default is 1.
    check_route : bool, optional
        Whether the snow event polygons should be avoided or not when generating the route. 
        The default is True.

    Returns
    -------
    SnowDirections object or tuple
        If routing is successful, the SnowDirections object is returned. Else a tuple with
        reasoning for failure is returned. 
    '''
    
    route = SnowDirections(snow_df, grocery_df, crime_df, 
                      grocery_name, home_address, 
                      radius, polygon_scale)
    
    route.geolocate_address()
    
    if not route.in_minneapolis:
        return (False,
                '''The address provided is not within Minneapolis, MN city limits.''',
                '''Please try again.''')
    
    route.data_trim()
    
    route.store_pref()
    if not route.store_matches:
        return (False, 
                '''The Grocery Store provided is not in the given radius. 
                Here is a list of stores in the radius:''', 
                route.groc['Name'].unique())
    
    route.cluster()
    route.snow_clust_df()
    route.crim_clust_df()
    route.rank_stores()
    
    route.poly_scale()
    route.find_route()
    if check_route == True:
        route.check_route()
    
    return route

#%%
#Plot for crime events per month to show that we should consider car break-ins and theft
#when ranking grocery stores from least to most safe.

#Filter out 2023 data, it's not complete as of Nov 2023.
crime_df = crime_df[crime_df['Year'] < 2023]

def plot_crime():
    '''Creates a plot of auto crime data per month in Minneapolis, MN from 2013-2022
    from crime_df data.'''
    
    averg = sum(crime_df.groupby('Month').count()['Year'])/12.
    minimum = crime_df.groupby('Month').count()['Year'].min()
    min_perc_diff = abs(averg-minimum)/averg*100
    
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cool = '#2ff5ee'
    warm = '#d9dede'
    p = [cool] * 4 + [warm] * 6 + [cool] * 2
    
    g = sns.countplot(x='Month', data=crime_df, palette=p)
    plt.plot([-1,13], [averg, averg], ls='--', color='black')
    plt.plot([3,3], [averg, minimum], ls=':', color='black')
    plt.xlim(-1,12)
    plt.ylim(0,7000)
    g.set_xticklabels(labels)
    g.set_title(r'Auto theft & break-ins per month (2013-2022)')
    g.set_ylabel(r'Count')
    
    ax = plt.gca()
    ax.text(0.5, 6000, rf'Monthly avg = {averg:.0f}')
    ax.text(2.2, 5200, rf'{min_perc_diff:.0f}%')
    ax.text(9.5, 6700, r'Snowy months', color = '#21bfba')
    ax.text(9.5, 6300, r'Warm months', color = 'grey')
    
    fig = plt.gcf()
    fig.set_size_inches(7,7)
    return fig