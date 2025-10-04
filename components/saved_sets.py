import streamlit as st
from datetime import datetime
from utils.database import (
    init_database,
    save_comparison_set,
    get_all_comparison_sets,
    delete_comparison_set,
    update_comparison_set
)

def render_saved_sets_manager():
    """Render the saved comparison sets manager in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Saved Comparison Sets")
    
    # Initialize database on first run
    if 'db_initialized' not in st.session_state:
        if init_database():
            st.session_state.db_initialized = True
        else:
            st.sidebar.error("Failed to initialize database")
            return
    
    # Get current selection
    current_companies = st.session_state.get('selected_companies', [])
    
    # Save current selection
    if current_companies and len(current_companies) >= 2:
        with st.sidebar.expander("💾 Save Current Set"):
            save_name = st.text_input(
                "Set Name",
                key="save_set_name",
                placeholder="e.g., Tech Giants Comparison"
            )
            
            save_description = st.text_area(
                "Description (optional)",
                key="save_set_description",
                placeholder="Brief description of this comparison set"
            )
            
            if st.button("Save Set", key="save_set_btn"):
                if save_name:
                    if save_comparison_set(save_name, save_description, current_companies):
                        st.success(f"Saved '{save_name}' successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to save comparison set")
                else:
                    st.warning("Please enter a name for the set")
    
    # Load saved sets
    saved_sets = get_all_comparison_sets()
    
    if saved_sets:
        st.sidebar.markdown("### 📂 Saved Sets")
        
        for saved_set in saved_sets:
            with st.sidebar.expander(f"{saved_set['name']} ({len(saved_set['companies'])} companies)"):
                st.markdown(f"**Description:** {saved_set['description'] or 'No description'}")
                st.markdown(f"**Companies:** {', '.join(saved_set['companies'])}")
                st.markdown(f"**Created:** {saved_set['created_at'].strftime('%Y-%m-%d %H:%M')}")
                
                # Use vertical layout instead of columns in sidebar
                if st.button("📥 Load Set", key=f"load_{saved_set['id']}", use_container_width=True):
                    st.session_state.selected_companies = saved_set['companies'].copy()
                    st.success(f"Loaded '{saved_set['name']}'")
                    st.rerun()
                
                if st.button("🗑️ Delete Set", key=f"delete_{saved_set['id']}", use_container_width=True):
                    if delete_comparison_set(saved_set['id']):
                        st.success(f"Deleted '{saved_set['name']}'")
                        st.rerun()
                    else:
                        st.error("Failed to delete")
    else:
        st.sidebar.info("No saved comparison sets yet. Save your current selection to get started!")

def render_saved_sets_page(comparison_data):
    """Render a detailed view of saved comparison sets"""
    st.subheader("💾 Saved Comparison Sets")
    
    # Initialize database
    if 'db_initialized' not in st.session_state:
        if init_database():
            st.session_state.db_initialized = True
        else:
            st.error("Failed to initialize database")
            return
    
    # Get all saved sets
    saved_sets = get_all_comparison_sets()
    
    if not saved_sets:
        st.info("No saved comparison sets yet. Save your current comparison to get started!")
        
        # Quick save section
        if comparison_data and len(comparison_data) >= 2:
            st.markdown("### Quick Save Current Comparison")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                quick_name = st.text_input(
                    "Set Name",
                    placeholder="e.g., Tech Giants Q4 2024"
                )
            
            with col2:
                if st.button("💾 Save", type="primary"):
                    if quick_name:
                        companies = list(comparison_data.keys())
                        if save_comparison_set(quick_name, "", companies):
                            st.success(f"Saved '{quick_name}'!")
                            st.rerun()
                    else:
                        st.warning("Please enter a name")
        return
    
    # Display saved sets in cards
    st.markdown("### Your Saved Sets")
    
    for i, saved_set in enumerate(saved_sets):
        with st.container():
            st.markdown(f"#### {saved_set['name']}")
            
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"**Companies:** {', '.join(saved_set['companies'])}")
                if saved_set['description']:
                    st.markdown(f"**Description:** {saved_set['description']}")
            
            with col2:
                st.markdown(f"**Created:** {saved_set['created_at'].strftime('%Y-%m-%d')}")
                st.markdown(f"**Updated:** {saved_set['updated_at'].strftime('%Y-%m-%d')}")
            
            with col3:
                if st.button("📥 Load", key=f"load_detail_{saved_set['id']}"):
                    st.session_state.selected_companies = saved_set['companies'].copy()
                    st.success("Loaded! Switch to Current Comparison tab.")
                    st.rerun()
                
                if st.button("🗑️ Delete", key=f"delete_detail_{saved_set['id']}"):
                    if delete_comparison_set(saved_set['id']):
                        st.success("Deleted!")
                        st.rerun()
            
            st.markdown("---")
    
    # Add new set section
    if comparison_data and len(comparison_data) >= 2:
        st.markdown("### Save Current Comparison")
        
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            new_name = st.text_input(
                "Set Name",
                key="new_set_name",
                placeholder="e.g., Banking Sector 2024"
            )
        
        with col2:
            new_description = st.text_input(
                "Description (optional)",
                key="new_set_description",
                placeholder="Brief description"
            )
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Save Set", type="primary"):
                if new_name:
                    companies = list(comparison_data.keys())
                    if save_comparison_set(new_name, new_description, companies):
                        st.success(f"Saved '{new_name}'!")
                        st.rerun()
                else:
                    st.warning("Please enter a name")
    
    # Statistics
    st.markdown("### Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Saved Sets", len(saved_sets))
    
    with col2:
        total_companies = len(set([company for s in saved_sets for company in s['companies']]))
        st.metric("Unique Companies", total_companies)
    
    with col3:
        if saved_sets:
            avg_companies = sum(len(s['companies']) for s in saved_sets) / len(saved_sets)
            st.metric("Avg. Companies per Set", f"{avg_companies:.1f}")
