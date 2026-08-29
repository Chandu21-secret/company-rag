import pandas as pd


def extract_dealers_from_excel(excel_path):

    df = pd.read_excel(
        excel_path,
        sheet_name="All India Dealers"
    )

    # Remove extra spaces from column names
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    dealer_records = []

    for _, row in df.iterrows():

        dealer_name = str(
            row.get("Dealer Name", "")
        ).strip()

        gst_no = str(
            row.get("GST No.", "")
        ).strip()

        address = str(
            row.get("Address", "")
        ).strip()

        district = str(
            row.get("District", "")
        ).strip()

        # Your Excel has State under "Unnamed: 5"
        state = str(
            row.get("Unnamed: 5", "")
        ).strip()

        pin_code = str(
            row.get("Pin Code", "")
        ).strip()

        region = str(
            row.get("Region", "")
        ).strip()

        rsm = str(
            row.get("RSM /AM/ SE", "")
        ).strip()

        contact_person = str(
            row.get("Contact Person", "")
        ).strip()

        mobile = str(
            row.get("Mobile No.", "")
        ).strip()

        email = str(
            row.get("Email Id", "")
        ).strip()

        # Skip empty rows
        if (
            not dealer_name
            or dealer_name.lower() == "nan"
        ):
            continue

        # Fix pandas NaN values
        if mobile.lower() == "nan":
            mobile = ""

        if email.lower() == "nan":
            email = ""

        if state.lower() == "nan":
            state = ""

        if district.lower() == "nan":
            district = ""

        if contact_person.lower() == "nan":
            contact_person = ""

        if rsm.lower() == "nan":
            rsm = ""

        text = f"""
Dealer Name: {dealer_name}
GST No.: {gst_no}
Address: {address}
District: {district}
State: {state}
Pin Code: {pin_code}
Region: {region}
RSM / AM / SE: {rsm}
Contact Person: {contact_person}
Mobile No.: {mobile}
Email Id: {email}
""".strip()

        dealer_records.append({

            "text": text,

            "source": "Dealers Database.xlsx",

            "page": None,

            "category": "dealer",

            "model": None,

            "dealer_name": dealer_name,

            "gst_no": gst_no,

            "address": address,

            "district": district,

            "state": state,

            "pin_code": pin_code,

            "region": region,

            "rsm": rsm,

            "contact_person": contact_person,

            "mobile": mobile,

            "email": email
        })

    return dealer_records