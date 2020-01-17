import numpy as np
import unittest
import os
import string

from ds_behavioral import DataBuilderComponent


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.name='test_build'
        if not os.path.exists('../simulators/data'):
            os.mkdir('../simulators/data')

    def tearDown(self):
        _tmp = DataBuilderComponent(self.name).fbpm
        _tmp.remove(_tmp.KEY.contract_key)
        try:
            os.remove('config_data_builder.yaml')
            # os.remove('customer.csv')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        DataBuilderComponent(self.name)

    def test_case(self):
        fb = DataBuilderComponent('case')
        # clear out the previous configuration
        _ = fb.fbpm.remove(fb.fbpm.KEY.contract_key)
        fb.fbpm.save()
        account_ids = fb.tools.unique_identifiers(from_value=1000000, to_value=9999999, seed=101, size=100)
        portfolio_ids = fb.tools.unique_identifiers(from_value=1000, to_value=9999, seed=101, size=100)
        ctp_ids = fb.tools.unique_identifiers(from_value=100, to_value=999, seed=101, size=10)
        # main columns
        fb.add_column('case_id', 'unique_identifiers', from_value=1000000, to_value=9999999)
        fb.add_column('account_id', 'get_category', selection=account_ids)
        fb.add_column('date_created', 'get_datetime', start='01/01/2015', until='29/11/2016', date_format='%d-%m-%Y')
        fb.add_column('date_closed', 'get_datetime', start='29/11/2016', until='29/11/2018', date_format='%d-%m-%Y')
        fb.add_column('portfolio_id', 'get_category', selection=portfolio_ids, quantity=0.1)
        fb.add_column('age_of_debt', 'get_number', to_value=125)
        fb.add_column('arrears', 'get_distribution', method='beta', offset=np.random.randint(1000, 7000), precision=2, a=2,
                      b=5)
        fb.add_column('broken_arrangement', 'get_category', selection=[True, False], weight_pattern=[0.1, 0.9], quantity=0.6)
        fb.add_column('date broken', 'get_datetime', start='29/11/2016', until='29/11/2018', date_format='%d-%m-%Y')
        fb.add_column('date_plan_agreed', 'get_datetime', start='29/11/2016', until='29/11/2018', date_format='%d-%m-%Y')
        fb.add_column('authorised_party', 'get_category', selection=ctp_ids)
        fb.add_column('authorised_date', 'get_datetime', start='29/11/2016', until='29/11/2018', date_format='%d-%m-%Y')
        fb.add_column('exception', 'get_category', selection=[True, False], weight_pattern=[0.02, 0.98], quantity=0.6)
        fb.add_column('reason_for_arrears', 'get_category', selection=[''], quantity=0.1)
        fb.add_column('reason_for_collection', 'get_category', selection=[''], quantity=0.01)
        fb.add_column('arrangement', 'get_category', selection=[''], quantity=0.01)
        fb.add_column('due_date', 'get_datetime', start='29/11/2016', until='29/11/2018', date_format='%d-%m-%Y')
        fb.add_column('debt_type', 'get_category', selection=['auto-finance', 'credit card', 'personal loan', 'telecoms'])
        fb.add_column('case_status', 'get_category', selection=['Sold', 'Open', 'Allocated'])
        fb.add_column('case_status_str', 'get_category', selection=[''], quantity=0.01)
        fb.add_column('dca_allocation', 'get_datetime', start='29/11/2017', until='29/11/2018', date_format='%d-%m-%Y',
                      quantity=0.4)
        fb.add_column('ctp_id', 'get_category', selection=ctp_ids)
        fb.add_column('recovery', 'get_distribution', method='beta', offset=np.random.randint(10, 7000), precision=2, a=2,
                      b=5)
        fb.add_column('amount_at_risk', 'get_distribution', method='beta', offset=np.random.randint(1000, 7000),
                      precision=2, a=2, b=5)
        fb.add_column('segmentation', 'get_category',
                      selection=['deceased', 'fraud', 'suspended', 'hardship', 'holdout', 'LR Cycle 1', 'LR Cycle 2',
                                 'MR Cycle 1', 'MR Cycle 2', 'HR Cycle 1', 'HR Cycle 2'], weight_pattern=[0.3, 0.1, 0.2, 0.2],
                      quantity=0.2)
        fb.fbpm.save()
        fb.build_columns(rows=1000, filename='../simulators/data/comms.csv')

    def test_Comms(self):
        fb = DataBuilderComponent('Comms')
        # clear out the previous configuration
        _ = fb.fbpm.remove(fb.fbpm.KEY.contract_key)
        fb.fbpm.save()
        # create Account id's
        account_ids = fb.tools.unique_identifiers(from_value=1000000, to_value=9999999, seed=101, size=100)
        # main columns
        fb.add_column('comms_id', 'unique_identifiers', from_value=1000000, to_value=9999999)
        fb.add_column('account_id', 'get_category', selection=account_ids)
        fb.add_column('date', 'get_datetime', start='01/01/2015 00:00', until='29/11/2018 23:59',
                      date_format='%d-%m-%Y %H:%M')
        fb.add_column('advisor_id', 'get_number', from_value=100, to_value=150)
        fb.add_column('reason', 'get_category', selection=fb.samples.call_centre.complaint)
        fb.add_column('secondary_reason', 'get_category', selection=fb.samples.call_centre.complaint, quantity=0.4)
        fb.add_column('call_type', 'get_category', selection=fb.samples.call_centre.contact_type, weight_pattern=[0.4, 0.2, 0.1])
        fb.add_column('doc_shared', 'get_category', selection=[True, False], weight_pattern=[0.02, 0.98], quantity=0.6)
        fb.add_column('arrangement', 'get_category', selection=['Pay', 'Referred', 'Standing Order'])
        fb.add_column('arrangement_notes', 'get_category', selection=[''])
        fb.add_column('channel', 'get_category', selection=fb.samples.call_centre.contact_type, weight_pattern=[0.4, 0.2, 0.1])
        fb.add_column('channel_reference', 'get_category', selection=[''])
        fb.add_column('complaint', 'get_category', selection=[True, False], weight_pattern=[0.02, 0.98], quantity=0.6)
        fb.add_column('outcome', 'get_category', selection=['Successful', 'Pending', 'Dissatisfied'],
                      weight_pattern=[0.8, 0.1, 0.1], quantity=0.8)
        fb.add_column('progressed', 'get_category', selection=[True, False], weight_pattern=[0.1, 0.9], quantity=0.6)
        fb.add_column('status', 'get_category', selection=['Open', 'Closed'], weight_pattern=[0.02, 0.98], quantity=0.8)
        fb.add_column('follow_up', 'get_category', selection=[True, False], weight_pattern=[0.1, 0.9], quantity=0.6)
        fb.add_column('notes', 'get_category', selection=fb.tools.unique_str_tokens(0, 0))
        fb.add_column('supervisor_called', 'get_category', selection=[True, False], weight_pattern=[0.02, 0.98],
                      quantity=0.6)
        fb.add_column('time_start', 'get_datetime', start='8:00:0000', until='23:00:000', date_format='%H:%M:%S')
        fb.add_column('time_end', 'get_datetime', start='8:00:0000', until='23:00:000', date_format='%H:%M:%S')
        fb.fbpm.save()
        fb.build_columns(rows=1000, filename='../simulators/data/comms.csv')

    def test_transaction(self):
        fb = DataBuilderComponent('Account')
        # clear out the previous configuration
        _ = fb.fbpm.remove(fb.fbpm.KEY.contract_key)
        fb.fbpm.save()
        # create Account id's
        account_ids = fb.tools.unique_identifiers(from_value=1000000, to_value=9999999, seed=101, size=100)
        # main columns
        fb.add_column('transaction_id', 'unique_identifiers', from_value=1000000, to_value=9999999)
        fb.add_column('account', 'get_category', selection=account_ids)
        fb.add_column('txn_date', 'unique_date_seq', start='01/01/2015', until='29/11/2018', date_format='%d-%m-%Y')
        fb.add_column('txn_type', 'get_category', selection=['credit', 'debit', 'one-off'], weight_pattern=[0.6, 0.3, 0.1])
        fb.add_column('txn_amount', 'get_distribution', method='beta', offset=np.random.randint(30, 300), precision=2,
                      a=2, b=5)
        fb.add_column('currency', 'get_category', selection=['GBP'])
        fb.add_column('txn_comments', 'get_category', selection=[np.nan])
        fb.add_column('txn_reference', 'get_category', selection=[np.nan])
        fb.fbpm.save()
        fb.build_columns(rows=1000, filename='../simulators/data/transaction.csv')


    def test_account(self):
        fb = DataBuilderComponent('Account')
        # clear out the previous configuration
        _ = fb.fbpm.remove(fb.fbpm.KEY.contract_key)
        fb.fbpm.save()
        # create Custonmer id's
        cids = fb.tools.unique_identifiers(from_value=100000, to_value=999999, prefix="CU_", seed=101, size=100)
        # main columns
        fb.add_column('account_id', 'unique_identifiers', from_value=1000000, to_value=9999999)
        fb.add_column('cid', 'get_category', selection=cids)
        fb.add_column('account_type', 'get_category',
                      selection=['credit card', 'loan', 'mortgage', 'rental', 'utilities'],
                      weight_pattern=[0.4, 0.2, 0.2, 0.1, 0.1])
        fb.add_column('account_status', 'get_category', selection=['Suspended', 'live', 'closed'],
                      weight_pattern=[0.1, 0.8, 0.1], quantity=0.7)
        fb.add_column('account_product', 'get_category', selection=['Amex Credit Card', 'iPhone instalments'],
                      quantity=0.3)
        fb.add_column('payment_type', 'get_category', selection=['monthly', 'Adhoc'])
        fb.add_column('block_code_1', 'get_category', selection=['soft block', 'hard block'], quantity=0.05)
        fb.add_column('block_code_2', 'get_category', selection=['legal'], quantity=0.2)
        fb.add_column('start_date', 'get_datetime', start='01/01/1964', until='01/01/2017', date_format='%d-%m-%Y',
                      date_pattern=[1, 1, 1, 2, 3, 4, 5, 10, 7, 7, 3, 3, 2])
        fb.add_column('last_payment', 'get_datetime', start='01/02/2017', until='10/11/2018', date_format='%d-%m-%Y',
                      date_pattern=[1, 1, 5, 20, 16, 5, 3, 1, 1, 1])
        fb.add_column('promises_made', 'get_category', selection=[0, 1, 2, 3, 4, 5],
                      weight_pattern=[0.05, 0.2, 0.3, 0.3, 0.1, 0.05], quantity=0.9)
        fb.add_column('complaint', 'get_category', selection=[True, False], weight_pattern=[0.02, 0.98], quantity=0.7)
        fb.add_column('annual_equvalent_rate', 'get_distribution', method='beta', precision=2, a=5, b=2)
        fb.add_column('current_oustanding_balance', 'get_distribution', method='beta',
                      offset=np.random.randint(100, 7000), precision=2, a=5, b=2)
        fb.add_column('payment_amount', 'get_distribution', method='beta', offset=np.random.randint(30, 300),
                      precision=2, a=5, b=2)
        fb.fbpm.save()
        fb.build_columns(rows=1000, filename='../simulators/data/account.csv')

    def test_customer_build(self):
        fb = DataBuilderComponent('Customer')
        # clear out the previous configuration
        _ = fb.fbpm.remove(fb.fbpm.KEY.contract_key)
        fb.fbpm.save()
        # main columns
        fb.add_column('cid', 'unique_identifiers', from_value=100000, to_value=999999,
                      prefix="{}_".format(fb.tools.unique_str_tokens(2, 2, pool=string.ascii_uppercase)))
        fb.add_column('postcode', 'get_string_pattern', pattern='UUddsdUU')
        fb.add_column('date_of_birth', 'get_datetime', start='01-01-1940', until='01-01-2000',
                      date_pattern=[1, 1, 8, 2, 3, 5, 9, 3])
        fb.add_column('vulnerability', 'get_category', selection=[True, False], weight_pattern=[0.05, 0.95], quantity=0.75)
        fb.add_column('fraud', 'get_category', selection=[True, False], weight_pattern=[0.05, 0.95], quantity=0.75)
        fb.add_column('deceased', 'get_category', selection=[True, False], weight_pattern=[0.02, 0.98], quantity=0.1)
        fb.add_column('gone_away', 'get_category', selection=[True, False], weight_pattern=[0.05, 0.95], quantity=0.1)
        fb.add_column('insolvency', 'get_category', selection=[True, False], weight_pattern=[0.02, 0.98], quantity=0.3)
        fb.add_column('employed', 'get_category', selection=[True, False], weight_pattern=[0.95, 0.05], quantity=0.80)
        fb.add_column('dependants', 'get_category', selection=[0, 1, 2, 3, 4, 5],
                      weight_pattern=[0.25, 0.4, 0.2, 0.1, 0.04, 0.01], quantity=0.80)
        fb.add_column('profession', 'get_category', selection=fb.samples.generic.profession[:15], quantity=0.80)
        # float frequency
        label_offset = {'wage': (100, 1000),
                        'child_benefit': (50, 200),
                        'rent_or_board_received': (100, 300),
                        'state_pension': (5, 50)
                        }
        for label, offset in label_offset.items():
            feq_label = "{}_frequency".format(label)
            fb.add_column(feq_label, 'get_category', selection=['Daily', 'Weekly', 'Monthly', 'Annually'],
                          weight_pattern=[0.04, 0.30, 0.65, 0.01], quantity=0.8)
            fb.add_column(label, 'get_distribution', method='beta', offset=np.random.randint(offset[0], offset[1]),
                          precision=2, a=5, b=2, quantity=0.8)
        # bool frequency
        for label in ['partner_wage', 'part_time_wages', 'private_pension']:
            feq_label = "{}_frequency".format(label)
            fb.add_column(feq_label, 'get_category', selection=['Daily', 'Weekly', 'Monthly', 'Annually'],
                          weight_pattern=[0.04, 0.30, 0.65, 0.01], quantity=0.8)
            fb.add_column(label, 'get_category', selection=[True, False], weight_pattern=[0.3, 0.7], quantity=0.8)
        # float values
        label_offset = {'pension_credit': (0, 100),
                        'employment_and_support_allowance': (0, 40),
                        'jobseeker_allowance': (0, 100),
                        'child_support_allowance': (0, 400),
                        'disability_living_allowance': (0, 400),
                        'income_support': (0, 200),
                        'working_tax_credit': (0, 50),
                        'child_tax_credit': (0, 20),
                        'housing_benefit': (0, 300),
                        'total_outgoings_housing_and_utilities': (200, 800),
                        'gas_payments': (40, 100),
                        'electricity_payments': (20, 60),
                        'water_payments': (10, 30),
                        'mortgage_payments': (500, 1700),
                        'rent_payments': (400, 1500),
                        'telephone_internet_payments': (10, 50),
                        'tv_license_payments': (9, 20),
                        'buildings_and_content_insurance_payments': (10, 100),
                        'council_tax': (100, 350),
                        'travel': (0, 1000),
                        'mobile_phone': (10, 60),
                        'household_services': (30, 60),
                        'food_and_housekeeping': (60, 400),
                        'other_services': (0, 60),
                        'personal_and_leisure': (50, 1000),
                        'other_costs': (0, 100),
                        'loans': (0, 4000),
                        'credit_cards': (100, 1000),
                        }
        for label, offset in label_offset.items():
            fb.add_column(label, 'get_distribution', method='beta', offset=np.random.randint(offset[0], offset[1]),
                          precision=2, a=5, b=2)
        fb.fbpm.save()
        fb.build_columns(rows=101, filename='../simulators/data/customer.csv')

if __name__ == '__main__':
    unittest.main()
