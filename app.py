import streamlit as st 
from streamlit_folium import st_folium
import snow_directions as sdir


st.set_page_config(page_title='Directions in a blizzard')

st.title('Groceries in a blizzard - Minneapolis, MN')

st.write('''Minneapolis, MN receives a ton of snow every year. 
         While most residents will try to remain at home during a snow storm,
         there will be instances where they need to go out, get in their car,
         and drive to a nearby grocery store to resupply.
         The goal of this project is to determine the safest route a resident
         can take to a nearby grocery store in a snow storm.''')

st.write('''As a proxy to how likely roads in an area will be cleared of snow and are safe to drive on,
         the algorithm clusters car tagging and towing events during snow storms provided by
         [Open Data Minneapolis](https://opendata.minneapolismn.gov/).
         Cars that are tagged have likely either parked in a way that prevents snow plows from 
         efficently clearing roads 
         ([Minneapolis parking rules in snow emergencies](https://www.minneapolismn.gov/getting-around/snow/snow-emergencies/snow-parking-rules/))
         or have been immobilized due to buildup of snow. These tagging events, and of course accidents,
         can lead to having a car towed. It's best to stick to roads that are clear!''')

st.write('''These tagging and towing clusters of events are then input into a routing algorithm 
         that utilizes the [openroute service API](https://openrouteservice.org).''')
         
st.write('[Code repository on GitHub](https://github.com/mitchellmb)')

st.header('Route generation')

st.write('''Input a grocery store you\'d like to visit, 
         your starting home address, 
         and a maximum distance you\'re willing to travel as a radius around your home.
         Note: Snow clusters ARE NOT avoided by default.''')


#User input section:   
groc = st.text_input('Grocery store', 'Target')
addr = st.text_input('Home address', '420 SE Main St, Minneapolis, MN 55414')
rad = st.slider('Radius (miles)', 1, 5, value=2, step=1)

st.write('Does the route differ if snow clusters are avoided?')
check = st.toggle('Avoid snow clusters', value=False)


#Route generation section:
@st.cache_resource
def load_directions(grocery, radius, address, check):
    #Cache directions so that the API isn't repeatedly called
    route = sdir.directions(groc, addr, radius=rad, polygon_scale=1, check_route=check)
    return route

route = load_directions(groc, rad, addr, check)


#If the route failed, print out why
if isinstance(route, tuple):
    st.subheader('Routing failed')
    st.write(route[1])
    st.write(route[2])

#If the route worked, print out results
else:
    st.subheader('Directions map')
    snow = st.toggle('Snow cluster centers', value=True)
    crime = st.toggle('Crime cluster centers', value=False)
    
    plot = route.plotting(crime_clust=crime, snow_clust=snow, poly_plot=False)
    st_folium(plot, width=700)  
    
    st.subheader(f'Directions to {groc}')
    
    
    if isinstance(route.instructions, list):
        instructions = route.instructions
    else:
        instructions = route.instructions()
    
    for i in instructions:
        st.markdown('- ' + f':white[{i}]')
    
    
    st.subheader('Alternative stores to consider')
    
    st.write('''While the above route is the best route to your preferred
             grocery store that avoids roads with high rates of car tagging and towing events,
             there may be a closer and safer nearby store you can go to instead.
             The top 10 safest & closest stores are listed below.''')
             
    st.write('''Perhaps try routing to one of these if your preferred store is not listed!''')
    #List the top 10 safest stores
    for i, name in enumerate(route.groc['Name'].iloc[0:10]):
        st.markdown('(' + str(i+1) + ') ' + name)
    
    st.write('''Grocery stores are ranked by how far they are from your home address
             and by rates of car tagging/towing events combined with the rates of 
             car theft/break-in events in the area.
             It may be counterintuitive, but extreme cold weather does not always stop
             auto theft and break-ins, and nobody wants to be able to get to a grocery store 
             in snowy weather only to find their method of returning home compromised! 
             You can view the rates of these crimes per month
             in Minneapolis below. This plot shows that while auto crimes do decrease
             in winter months (maximum 15% drop from the average in April), 
             they do not halt altogether.'''
             )

    st.pyplot(sdir.plot_crime())
