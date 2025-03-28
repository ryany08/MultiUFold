from arnie.pk_predictors import _hungarian
from main.utils import mask_diagonal, contact_2_dotbracket_and_bplist
from utils.metrics import calculate_f1_score_with_pseudoknots

def post_process(contact_pred, contact_true, length):
    bps_pred, _ = contact_2_dotbracket_and_bplist(contact_pred, length, isSigmoid=True)
    bps_true, _ = contact_2_dotbracket_and_bplist(contact_true, length, isSigmoid=False)
    precision, recall, f1_score = calculate_f1_score_with_pseudoknots(bps_true, bps_pred)
    return precision, recall, f1_score
