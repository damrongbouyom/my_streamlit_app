import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Mock Data Generation ---
def generate_mock_data():
    np.random.seed(42)
    
    # Dates for the current and previous year
    end_date_current = datetime.now()
    start_date_current = end_date_current - timedelta(days=30) # Last month
    dates_current = pd.date_range(start=start_date_current, end=end_date_current, freq='D')

    start_date_prev = start_date_current.replace(year=start_date_current.year - 1)
    end_date_prev = end_date_current.replace(year=end_date_current.year - 1)
    dates_prev = pd.date_range(start=start_date_prev, end=end_date_prev, freq='D')

    # Stores and Regions
    stores = [f'Store {i}' for i in range(1, 11)]
    regions = ['North', 'South', 'East', 'West']
    
    # Products and Categories
    product_categories = ['Electronics', 'Groceries', 'Apparel', 'Home Goods']
    products = {
        'Electronics': [f'Product E{i}' for i in range(1, 6)],
        'Groceries': [f'Product G{i}' for i in range(1, 6)],
        'Apparel': [f'Product A{i}' for i in range(1, 6)],
        'Home Goods': [f'Product H{i}' for i in range(1, 6)],
    }

    data = []
    for date in dates_current:
        for _ in range(np.random.randint(50, 150)): # Number of orders per day
            store = np.random.choice(stores)
            region = np.random.choice(regions)
            is_member = np.random.choice([True, False], p=[0.6, 0.4])
            category = np.random.choice(product_categories)
            product = np.random.choice(products[category])
            quantity = np.random.randint(1, 5)
            price_per_unit = np.random.uniform(5, 200)
            
            data.append({
                'Date': date,
                'Store': store,
                'Region': region,
                'Is_Member': is_member,
                'Product_Category': category,
                'Product': product,
                'Quantity': quantity,
                'Price_Per_Unit': price_per_unit,
                'Total_Sales': quantity * price_per_unit,
                'Order_ID': f'ORD{date.strftime("%Y%m%d")}{np.random.randint(1000, 9999)}'
            })
    
    df_current = pd.DataFrame(data)

    # Previous year data (mimic current year patterns but with some variance)
    data_prev = []
    for date in dates_prev:
        for _ in range(np.random.randint(45, 140)): # Slightly different order count
            store = np.random.choice(stores)
            region = np.random.choice(regions)
            is_member = np.random.choice([True, False], p=[0.6, 0.4])
            category = np.random.choice(product_categories)
            product = np.random.choice(products[category])
            quantity = np.random.randint(1, 5)
            price_per_unit = np.random.uniform(4.5, 190) # Slightly different prices
            
            data_prev.append({
                'Date': date,
                'Store': store,
                'Region': region,
                'Is_Member': is_member,
                'Product_Category': category,
                'Product': product,
                'Quantity': quantity,
                'Price_Per_Unit': price_per_unit,
                'Total_Sales': quantity * price_per_unit,
                'Order_ID': f'ORD{date.strftime("%Y%m%d")}{np.random.randint(1000, 9999)}'
            })
    df_prev = pd.DataFrame(data_prev)
    
    # Combine and add 'Year' column
    df_current['Year'] = 'Current Year'
    df_prev['Year'] = 'Previous Year'
    
    df_combined = pd.concat([df_current, df_prev])
    
    # Get the latest month available from the current year data
    latest_month = df_current['Date'].dt.to_period('M').max()
    df_filtered = df_current[df_current['Date'].dt.to_period('M') == latest_month]
    
    return df_combined, df_filtered, latest_month

df_combined, df_latest_month, latest_month_period = generate_mock_data()

# --- Helper Functions for KPIs ---
def get_kpi_status(actual, target):
    if actual >= target * 1.05:
        return 'green'
    elif actual >= target * 0.95:
        return 'yellow'
    else:
        return 'red'

def get_kpi_color_code(status):
    if status == 'green':
        return '#28a745'
    elif status == 'yellow':
        return '#ffc107'
    else:
        return '#dc3545'

# --- Dashboard Layout ---
st.set_page_config(layout="wide", page_title="Supermarket Performance Dashboard")

st.title(f"Supermarket Performance Dashboard ({latest_month_period.strftime('%B %Y')})")

# --- Row 1: Total Sales ---
st.markdown("---")
st.header("Total Sales")
col1_r1, col2_r1, col3_r1 = st.columns(3)

with col1_r1:
    st.subheader("Overall Sales Performance")
    total_sales_current_month = df_latest_month['Total_Sales'].sum()
    total_sales_target = 1000000 + np.random.randint(-50000, 50000) # Mock target
    sales_status = get_kpi_status(total_sales_current_month, total_sales_target)
    sales_color = get_kpi_color_code(sales_status)
    
    st.markdown(f"""
    <div style="
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 5px;
        background-color: #f9f9f9;
        text-align: center;
    ">
        <h3>Total Sales</h3>
        <h1 style="color: {sales_color};">${total_sales_current_month:,.2f}</h1>
        <p>Target: ${total_sales_target:,.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    # Line chart comparing current year vs. previous year sales
    sales_by_date = df_combined.groupby(['Date', 'Year'])['Total_Sales'].sum().reset_index()
    fig_sales_trend = px.line(sales_by_date, x='Date', y='Total_Sales', color='Year',
                              title='Total Sales: Current Year vs. Previous Year',
                              color_discrete_map={'Current Year': 'blue', 'Previous Year': 'lightgray'})
    fig_sales_trend.update_layout(height=250, margin=dict(t=50, b=0)) # Adjust height and margins
    st.plotly_chart(fig_sales_trend, use_container_width=True)

with col2_r1:
    st.subheader("Sales Breakdown by Region")
    sales_by_region = df_latest_month.groupby('Region')['Total_Sales'].sum().reset_index()
    fig_sales_region = px.bar(sales_by_region, x='Region', y='Total_Sales', 
                              title='Sales by Region', color='Region')
    fig_sales_region.update_layout(height=350, margin=dict(t=50, b=0))
    st.plotly_chart(fig_sales_region, use_container_width=True)

with col3_r1:
    st.subheader("AI-Generated Alerts (High-Risk Stores)")
    store_sales = df_latest_month.groupby('Store')['Total_Sales'].sum().reset_index()
    avg_sales_per_store = store_sales['Total_Sales'].mean()
    
    # Flag stores underperforming by a certain percentage
    underperforming_threshold = avg_sales_per_store * 0.8
    high_risk_stores = store_sales[store_sales['Total_Sales'] < underperforming_threshold].sort_values(by='Total_Sales')

    if not high_risk_stores.empty:
        st.error("⚠️ **High-Risk Stores (Underperforming in Total Sales):**")
        for index, row in high_risk_stores.iterrows():
            st.write(f"- **{row['Store']}**: Sales: ${row['Total_Sales']:,.2f} (Below average)")
    else:
        st.success("✅ No high-risk stores detected based on current sales performance.")

# --- Row 2: Number of Orders ---
st.markdown("---")
st.header("Number of Orders")
col1_r2, col2_r2, col3_r2 = st.columns(3)

with col1_r2:
    st.subheader("Overall Orders Performance")
    total_orders_current_month = df_latest_month['Order_ID'].nunique()
    total_orders_target = 5000 + np.random.randint(-500, 500) # Mock target
    orders_status = get_kpi_status(total_orders_current_month, total_orders_target)
    orders_color = get_kpi_color_code(orders_status)

    st.markdown(f"""
    <div style="
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 5px;
        background-color: #f9f9f9;
        text-align: center;
    ">
        <h3>Number of Orders</h3>
        <h1 style="color: {orders_color};">{total_orders_current_month:,}</h1>
        <p>Target: {total_orders_target:,}</p>
    </div>
    """, unsafe_allow_html=True)

    # Line chart comparing current year vs. previous year orders
    orders_by_date = df_combined.groupby(['Date', 'Year'])['Order_ID'].nunique().reset_index()
    fig_orders_trend = px.line(orders_by_date, x='Date', y='Order_ID', color='Year',
                              title='Number of Orders: Current Year vs. Previous Year',
                              labels={'Order_ID': 'Number of Orders'},
                              color_discrete_map={'Current Year': 'blue', 'Previous Year': 'lightgray'})
    fig_orders_trend.update_layout(height=250, margin=dict(t=50, b=0))
    st.plotly_chart(fig_orders_trend, use_container_width=True)

with col2_r2:
    st.subheader("Orders: Members vs. Non-Members")
    orders_by_membership = df_latest_month.groupby(['Date', 'Is_Member'])['Order_ID'].nunique().reset_index()
    orders_by_membership['Member_Type'] = orders_by_membership['Is_Member'].apply(lambda x: 'Member' if x else 'Non-Member')
    fig_member_orders = px.line(orders_by_membership, x='Date', y='Order_ID', color='Member_Type',
                                title='Orders from Members vs. Non-Members Over Time',
                                labels={'Order_ID': 'Number of Orders'},
                                color_discrete_map={'Member': 'purple', 'Non-Member': 'orange'})
    fig_member_orders.update_layout(height=350, margin=dict(t=50, b=0))
    st.plotly_chart(fig_member_orders, use_container_width=True)

with col3_r2:
    st.subheader("Order Breakdown by Product Category")
    # Simulate previous month for decline analysis
    # For a real scenario, you'd load actual previous month's data
    df_prev_month_simulated = df_combined[
        (df_combined['Date'].dt.to_period('M') == (latest_month_period - 1)) & 
        (df_combined['Year'] == 'Current Year') # Ensure we're comparing current year's prev month
    ]
    
    orders_current_cat = df_latest_month.groupby('Product_Category')['Order_ID'].nunique().reset_index()
    orders_current_cat.rename(columns={'Order_ID': 'Current_Month_Orders'}, inplace=True)

    orders_prev_cat_simulated = df_prev_month_simulated.groupby('Product_Category')['Order_ID'].nunique().reset_index()
    orders_prev_cat_simulated.rename(columns={'Order_ID': 'Previous_Month_Orders'}, inplace=True)
    
    category_orders = pd.merge(orders_current_cat, orders_prev_cat_simulated, on='Product_Category', how='left').fillna(0)
    category_orders['Orders_Decline'] = category_orders['Previous_Month_Orders'] - category_orders['Current_Month_Orders']
    category_orders['Orders_Decline_Percentage'] = (category_orders['Orders_Decline'] / category_orders['Previous_Month_Orders']) * 100
    
    # Highlight categories with largest declines
    top_declines = category_orders.sort_values(by='Orders_Decline', ascending=False).head(5)

    st.write("Categories with the largest declines in orders (vs. previous month):")
    st.dataframe(top_declines[['Product_Category', 'Current_Month_Orders', 'Previous_Month_Orders', 'Orders_Decline', 'Orders_Decline_Percentage']].style.format({
        'Current_Month_Orders': '{:,.0f}',
        'Previous_Month_Orders': '{:,.0f}',
        'Orders_Decline': '{:,.0f}',
        'Orders_Decline_Percentage': '{:,.2f}%'
    }), hide_index=True, use_container_width=True)


# --- Row 3: Ticket Size ---
st.markdown("---")
st.header("Ticket Size")
col1_r3, col2_r3, col3_r3 = st.columns(3)

with col1_r3:
    st.subheader("Average Ticket Size Performance")
    # Calculate average ticket size per order
    # Group by Order_ID to get total sales per order, then average that
    sales_per_order = df_latest_month.groupby('Order_ID')['Total_Sales'].sum()
    avg_ticket_size_current_month = sales_per_order.mean()

    avg_ticket_size_target = 150 + np.random.uniform(-10, 10) # Mock target
    ticket_status = get_kpi_status(avg_ticket_size_current_month, avg_ticket_size_target)
    ticket_color = get_kpi_color_code(ticket_status)

    st.markdown(f"""
    <div style="
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 5px;
        background-color: #f9f9f9;
        text-align: center;
    ">
        <h3>Average Ticket Size</h3>
        <h1 style="color: {ticket_color};">${avg_ticket_size_current_month:,.2f}</h1>
        <p>Target: ${avg_ticket_size_target:,.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    # Line chart comparing current year vs. previous year average ticket size
    avg_ticket_by_date = df_combined.groupby(['Date', 'Year', 'Order_ID'])['Total_Sales'].sum().reset_index()
    avg_ticket_by_date = avg_ticket_by_date.groupby(['Date', 'Year'])['Total_Sales'].mean().reset_index()
    avg_ticket_by_date.rename(columns={'Total_Sales': 'Average_Ticket_Size'}, inplace=True)
    
    fig_ticket_trend = px.line(avg_ticket_by_date, x='Date', y='Average_Ticket_Size', color='Year',
                              title='Average Ticket Size: Current Year vs. Previous Year',
                              color_discrete_map={'Current Year': 'blue', 'Previous Year': 'lightgray'})
    fig_ticket_trend.update_layout(height=250, margin=dict(t=50, b=0))
    st.plotly_chart(fig_ticket_trend, use_container_width=True)

with col2_r3:
    st.subheader("Ticket Size: Members vs. Non-Members")
    # Calculate average ticket size for members vs non-members
    avg_ticket_by_membership = df_latest_month.groupby(['Date', 'Is_Member', 'Order_ID'])['Total_Sales'].sum().reset_index()
    avg_ticket_by_membership = avg_ticket_by_membership.groupby(['Date', 'Is_Member'])['Total_Sales'].mean().reset_index()
    avg_ticket_by_membership['Member_Type'] = avg_ticket_by_membership['Is_Member'].apply(lambda x: 'Member' if x else 'Non-Member')
    avg_ticket_by_membership.rename(columns={'Total_Sales': 'Average_Ticket_Size'}, inplace=True)

    fig_member_ticket = px.line(avg_ticket_by_membership, x='Date', y='Average_Ticket_Size', color='Member_Type',
                                title='Average Ticket Size for Members vs. Non-Members',
                                color_discrete_map={'Member': 'purple', 'Non-Member': 'orange'})
    fig_member_ticket.update_layout(height=350, margin=dict(t=50, b=0))
    st.plotly_chart(fig_member_ticket, use_container_width=True)

with col3_r3:
    st.subheader("Ticket Size vs. Number of Orders by Store")
    store_performance = df_latest_month.groupby(['Store', 'Order_ID'])['Total_Sales'].sum().reset_index()
    store_performance_summary = store_performance.groupby('Store').agg(
        Average_Ticket_Size=('Total_Sales', 'mean'),
        Number_of_Orders=('Order_ID', 'nunique')
    ).reset_index()
    
    fig_scatter_store = px.scatter(store_performance_summary, 
                                   x='Number_of_Orders', 
                                   y='Average_Ticket_Size', 
                                   text='Store', 
                                   size_max=60,
                                   title='Ticket Size vs. Number of Orders by Store',
                                   labels={'Number_of_Orders': 'Number of Orders', 'Average_Ticket_Size': 'Average Ticket Size'})
    fig_scatter_store.update_traces(textposition='top center')
    fig_scatter_store.update_layout(height=350, margin=dict(t=50, b=0))
    st.plotly_chart(fig_scatter_store, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Generated with Mock Supermarket Data. For real-time analysis, integrate with actual data sources.")