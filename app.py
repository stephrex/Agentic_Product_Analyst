import streamlit as st
import asyncio
import json
import math
import plotly.express as px
import plotly.graph_objects as go

# Import your helper functions + ProductResearcher class
from app_functions import ProductResearcher, plot_wordcloud

# ---------------------------
# Streamlit App Setup
# ---------------------------
st.set_page_config(page_title="AI Product Analyst", layout="wide")

# Sidebar
st.sidebar.title("AI Product Analyst")
st.sidebar.markdown("""
This app is an **Agentic AI Analyst** that:
- Researches the internet for financial/business insights.  
- Analyzes the findings.  
- Returns insights **like a data analyst would**.  
- Creates **interactive visualizations** for clarity.  
""")

# Chat UI header
st.title("AI Product Analyst")

# Input box
user_input = st.chat_input("Type your research query here...")

# ---------------------------
# Function Mapper (Matplotlib args -> Plotly equivalents)
# ---------------------------


def map_to_plotly(func_name, args):
    fig = None

    if func_name == "plot_bar_chart_main":
        fig = px.bar(x=args.get("categories", args.get("labels", [])),
                     y=args.get("values", []),
                     title=args.get("title", "Bar Chart"),
                     labels={"x": args.get("xlabel", ""), "y": args.get("ylabel", "")},
                     color_discrete_sequence=[args.get("color", "skyblue")])

    elif func_name == "plot_stacked_bar_chart_main":
        df = {k: v for k, v in args.get("data", {}).items()}
        categories = args.get("categories", [])
        fig = go.Figure()
        for key, vals in df.items():
            fig.add_bar(name=key, x=categories, y=vals)
        fig.update_layout(barmode="stack", title=args.get("title", "Stacked Bar Chart"),
                          xaxis_title=args.get("xlabel", ""), yaxis_title=args.get("ylabel", ""))

    elif func_name == "plot_pie_chart_main":
        fig = px.pie(names=args.get("labels", []),
                     values=args.get("values", []),
                     title=args.get("title", "Pie Chart"))

    elif func_name == "plot_line_chart_main":
        fig = px.line(x=args.get("x", []), y=args.get("y", []),
                      title=args.get("title", "Line Chart"),
                      labels={"x": args.get("xlabel", ""), "y": args.get("ylabel", "")})

    elif func_name == "plot_area_chart_main":
        fig = px.area(x=args.get("x", []), y=args.get("y", []),
                      title=args.get("title", "Area Chart"),
                      labels={"x": args.get("xlabel", ""), "y": args.get("ylabel", "")})

    elif func_name == "plot_histogram_main":
        fig = px.histogram(x=args.get("values", []),
                           nbins=args.get("bins", 10),
                           title=args.get("title", "Histogram"),
                           labels={"x": args.get("xlabel", ""), "y": args.get("ylabel", "")})

    elif func_name == "plot_scatter_chart_main":
        fig = px.scatter(x=args.get("x", []), y=args.get("y", []),
                         title=args.get("title", "Scatter Plot"),
                         labels={"x": args.get("xlabel", ""), "y": args.get("ylabel", "")})

    elif func_name == "plot_box_chart_main":
        fig = px.box(y=args.get("values", []),
                     title=args.get("title", "Box Plot"),
                     labels={"y": args.get("ylabel", "Values")})

    elif func_name == "plot_heatmap_main":
        z = args.get("data", [])
        labels = args.get("labels", [])
        fig = px.imshow(z, labels=dict(x="Columns", y="Rows", color="Value"),
                        x=labels, y=labels, title=args.get("title", "Heatmap"),
                        color_continuous_scale=args.get("cmap", "Blues"))

    elif func_name == "plot_candlestick_chart_main":
        fig = go.Figure(data=[go.Candlestick(
            x=args.get("dates", []),
            open=args.get("open_prices", []),
            high=args.get("high_prices", []),
            low=args.get("low_prices", []),
            close=args.get("close_prices", []))])
        fig.update_layout(title=args.get("title", "Candlestick Chart"))

    elif func_name == "plot_moving_average_chart_main":
        dates = args.get("dates", [])
        values = args.get("values", [])
        window = args.get("window", 7)
        if len(values) >= window:
            ma = [None]*(window-1) + [sum(values[i-window+1:i+1])/window for i in range(window-1, len(values))]
        else:
            ma = values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", name="Raw"))
        fig.add_trace(go.Scatter(x=dates, y=ma, mode="lines", name=f"{window}-period MA"))
        fig.update_layout(title=args.get("title", "Moving Average Chart"),
                          xaxis_title=args.get("xlabel", ""), yaxis_title=args.get("ylabel", ""))

    elif func_name == "plot_stacked_area_chart_main":
        df = args.get("data", {})
        x = args.get("x", [])
        fig = go.Figure()
        for key, vals in df.items():
            fig.add_trace(go.Scatter(x=x, y=vals, stackgroup="one", name=key))
        fig.update_layout(title=args.get("title", "Stacked Area Chart"),
                          xaxis_title=args.get("xlabel", ""), yaxis_title=args.get("ylabel", ""))

    elif func_name == "plot_bubble_chart_main":
        fig = px.scatter(x=args.get("x", []), y=args.get("y", []),
                         size=args.get("sizes", []),
                         color_discrete_sequence=[args.get("color", "blue")],
                         title=args.get("title", "Bubble Chart"),
                         labels={"x": args.get("xlabel", ""), "y": args.get("ylabel", "")})

    elif func_name == "plot_donut_chart_main":
        fig = px.pie(names=args.get("labels", []),
                     values=args.get("values", []),
                     hole=0.4,
                     title=args.get("title", "Donut Chart"))

    else:
        fig = go.Figure()
        fig.add_annotation(text=f"{func_name} not yet mapped",
                           showarrow=False, x=0.5, y=0.5)

    return fig


# ---------------------------
# Handle user input
# ---------------------------
if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    # AI Response Placeholder
    with st.chat_message("assistant"):
        st.write("Running agentic AI research... Please wait.")
        placeholder = st.empty()

        # Run the researcher
        researcher = ProductResearcher()
        result, plot_funcs = asyncio.run(researcher.run(user_input))

        # --- Show AI textual output ---
        if result.final_output:
            placeholder.empty()  # remove loading
            st.markdown("### AI Findings")
            st.write(result.final_output)

        # --- Show Visualizations ---
        if len(plot_funcs) > 0:
            st.markdown("### Visualizations")
            n_plots = len(plot_funcs)
            ncols = 2
            nrows = math.ceil(n_plots / ncols)
            cols = st.columns(ncols)

            for i, (func, args) in enumerate(plot_funcs):
              if isinstance(args, str):
                  args = json.loads(args)
              
              if func.__name__ == "plot_word_cloud_main":
                # Word cloud is image-based, not Plotly. Show as text placeholder.
                words = args.get("text", "")
                buf = plot_wordcloud(words)
                st.image(buf, caption=args.get("title", "Word Cloud"))
                
              else:
                fig = map_to_plotly(func.__name__, args)

                with cols[i % ncols]:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True}, key=f"col_plot_{i}")
                    with st.expander("🔍 View Larger"):
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True}, key=f"expander_plot_{i}")

        else:
            st.info("No plots were generated by the AI.")
