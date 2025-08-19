import pandas as pd

def agri_import_partner_dependency(import_path, year=None, n_partners=3, threshold=0.6):
    import_df = pd.read_csv(import_path)
    print(import_df['cmdDesc'].dropna().unique())
    print("Columns in import_df:", import_df.columns.tolist())  # To help check columns

    if year is None:
        year = import_df['Year'].max()

    agri_keywords = [
        'crop', 'livestock', 'wheat', 'rice', 'corn', 'maize', 'soy', 'barley',
        'cattle', 'milk', 'all commodity', 'all commodities', 'commodity', 'commodities', 'all'
    ]
    agri_mask = import_df['cmdDesc'].str.lower().str.contains('|'.join(agri_keywords), na=False)
    agri_df = import_df[(import_df['Year'] == year) & agri_mask]

    results = []
    for country, grp in agri_df.groupby('Country'):
        total_imports = grp['import_value'].sum()
        partner_imports = grp.groupby('partnerDesc')['import_value'].sum().sort_values(ascending=False)
        top_partners = partner_imports.head(n_partners)
        top_share = top_partners.sum() / total_imports if total_imports > 0 else 0
        partner_names = list(top_partners.index)
        results.append({
            'Country': country,
            'TopPartners': partner_names,
            'TopPartnersImportSum': top_partners.sum(),
            'TotalAgriImports': total_imports,
            'TopPartnerShare': top_share
        })

    dep_df = pd.DataFrame(results)
    print("dep_df shape:", dep_df.shape)
    print("dep_df columns:", dep_df.columns.tolist())
    print(dep_df.head())

    if dep_df.empty or 'TopPartnerShare' not in dep_df.columns:
        print("No results found for given year and agricultural keywords.")
        return pd.DataFrame()

    most_dependent = dep_df[dep_df['TopPartnerShare'] >= threshold].sort_values('TopPartnerShare', ascending=False)

    return most_dependent


def model_food_security_risk(dep_df):
    dep_df = dep_df.copy()
    # Estimate import loss percentage if top partners impose export bans
    dep_df['ImportLoss_pct'] = dep_df['TopPartnerShare'] * 100
    # Flag countries with >40% import loss as high risk (arbitrary threshold)
    dep_df['RiskFlag'] = dep_df['ImportLoss_pct'] > 40
    return dep_df[['Country', 'TopPartners', 'TopPartnerShare', 'ImportLoss_pct', 'RiskFlag']]


