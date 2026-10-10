from config import FEATURES, RESPIRATORY, KIDNEY, MODERATE_THRESHOLD, HIGH_THRESHOLD

def availability(values, fields):
    count=sum(values[f] is not None for f in fields)
    return 'Available' if count==len(fields) else 'Insufficient' if count==0 else 'Partial'

def input_context(values):
    missing=[f for f in FEATURES if values[f] is None]
    warnings=[f'{f} is missing; training median used for model input' for f in missing]
    observed=len(FEATURES)-len(missing)
    # Presentation policy, not a validated adequacy threshold.
    quality='Insufficient clinical information' if observed<5 else 'Partial clinical information' if missing else 'Observed values available'
    if observed<5: warnings.insert(0,'Very sparse input: the estimate is predominantly based on imputed values and may be unreliable.')
    warnings.append('Missing does not mean normal. Model probability is uncalibrated and is not an individual clinical likelihood.')
    return dict(lung_status=availability(values,RESPIRATORY),kidney_status=availability(values,KIDNEY),
                data_quality=quality,observed_features=observed,missing_features=missing,warnings=warnings)

def risk_level(probability):
    return 'High' if probability>=HIGH_THRESHOLD else 'Moderate' if probability>=MODERATE_THRESHOLD else 'Low'
