import numpy as np
import logging 
import time 

logger = logging.getLogger(__name__)

from qecsim import paulitools as pt
from qecsim.app import _add_rate_statistics

from qcext.ptext import support 
import qcext # only require qcext.__version__ to stamp data, not circular. 

def _run_once_ft(code, time_steps, num_cycles, error_model, decoder, readout, error_probability, 
    measurement_error_probability, rng, results="simple", initial_error=None):
    """Implements run_once and run_once_ftp functions."""       

    # generate step_error, step_syndrome and step_measurement_error for each time step
    if initial_error is not None: 
        residual_error = initial_error 
    else: 
        residual_error = np.zeros(2*code.n_k_d[0], dtype=int) 

    if results == "full_history": 
        residual_error_history = np.zeros((num_cycles+1, 2*code.n_k_d[0]), dtype=int)
        residual_error_history[0,:] = residual_error

    for cycle in range(num_cycles):  

        step_errors, step_syndromes, step_measurement_errors = [], [], []

        # print("Residual error:") # REMOVE
        # print(code.ascii_art(pauli=code.new_pauli(bsf=residual_error))) # REMOVE
        for _ in range(time_steps):
            # step_error: random e rror based on error probability
            step_error = error_model.generate(code, error_probability, rng)
            step_errors.append(step_error)
            # step_syndrome: stabilizers that do not commute with the error
            step_syndrome = pt.bsp(step_error, code.stabilizers.T)
            step_syndromes.append(step_syndrome)
            residual_syndrome = pt.bsp(residual_error, code.stabilizers.T) 
            # step_measurement_error: random syndrome bit flips based on measurement_error_probability
            if measurement_error_probability:
                step_measurement_error = rng.choice(
                    (0, 1),
                    size=step_syndrome.shape,
                    p=(1 - measurement_error_probability, measurement_error_probability)
                ) 
            else:
                step_measurement_error = np.zeros(step_syndrome.shape, dtype=int)
            step_measurement_errors.append(step_measurement_error)

        if logger.isEnabledFor(logging.DEBUG): # for performance, perhaps?  
            try: 
                for i, step_error in enumerate(step_errors): 
                    logger.debug('run: step_error[{}]:\n{}'.format(i, code.ascii_art(pauli=code.new_pauli(bsf=step_error)))) 
                for i, step_syndrome in enumerate(step_syndromes): 
                    logger.debug("run: step_syndrome[{}]:\n{}".format(i, code.ascii_art(syndrome=step_syndrome)))  
                for i, step_measurement_error in enumerate(step_measurement_errors): 
                    logger.debug("run: step_measurement_error[{}]:\n{}".format(i, code.ascii_art(syndrome=step_measurement_error))) 
            except AttributeError: 
                logger.debug('run: step_errors={}'.format(step_errors))
                logger.debug('run: step_syndromes={}'.format(step_syndromes))
                logger.debug('run: step_measurement_errors={}'.format(step_measurement_errors))

        # error: sum of errors at each time step
        error = np.bitwise_xor.reduce([residual_error] + step_errors)

        if logger.isEnabledFor(logging.DEBUG):
            try: 
                logger.debug("run: total error:\n{}".format(code.ascii_art(pauli=code.new_pauli(bsf=error)))) 
            except AttributeError: 
                logger.debug('run: error={}'.format(error))

        # syndrome: apply measurement_error at times t-1 and t to syndrome at time t 
        step_measurement_errors.append(np.zeros(step_syndrome.shape, dtype=int)) # ensure smooth t=0 bc 
        syndrome = [] 
        for t in range(time_steps):
            syndrome.append(step_measurement_errors[t - 1] ^ step_syndromes[t] ^ step_measurement_errors[t])
        syndrome[0] = np.bitwise_xor.reduce([syndrome[0], residual_syndrome]) 
        # convert syndrome to 2d numpy array
        syndrome = np.array(syndrome)

        if logger.isEnabledFor(logging.DEBUG):
            try: 
                for i, syndrome_slice in enumerate(syndrome): 
                    logger.debug("run: syndrome[{}]: \n{}".format(i, code.ascii_art(syndrome=syndrome[i]))) 
            except AttributeError: 
                logger.debug('run: syndrome={}'.format(syndrome))

        # decoding: boolean or best match recovery operation based on decoder
        ctx = {'error_model': error_model, 'error_probability': error_probability, 'error': error,
               'step_errors': step_errors, 'measurement_error_probability': measurement_error_probability,
               'step_measurement_errors': step_measurement_errors, 'rng': rng} 
        
        recovery = decoder.decode_ft(code, time_steps, syndrome, **ctx) 
        residual_error = recovery ^ error

        if results == "full_history": 
            residual_error_history[cycle+1,:] = residual_error 

        if logger.isEnabledFor(logging.DEBUG):
            try: 
                logger.debug("run: recovery:\n{}".format(code.ascii_art(pauli=code.new_pauli(bsf=recovery)))) 
                logger.debug("run: residual_error:\n{}".format(code.ascii_art(pauli=code.new_pauli(bsf=residual_error)))) 
            except AttributeError: 
                logger.debug('run: recovery={}'.format(recovery))
                logger.debug("run: residual_error={}".format(residual_error)) 


    # success checks
    # compute readout error from measurement in the computational basis. 
    qubit_readout_error = readout.generate_error(code, measurement_error_probability, rng) 
    conserved_stabilisers_support = np.apply_along_axis(support, 1, readout.conserved_stabilisers(code)) 
    stabiliser_readout_error = np.dot(qubit_readout_error, conserved_stabilisers_support.T)%2
    readout_syndrome = pt.bsp(residual_error, readout.conserved_stabilisers(code).T) ^ stabiliser_readout_error 
    correction = readout.decode(code, readout_syndrome)

    included_readout_syndrome = np.zeros(np.shape(code.stabilizers)[0]) # for logging.  
    for s, stabiliser in zip(readout_syndrome, readout.conserved_stabilisers(code)): 
        index = np.where(np.all(code.stabilizers==stabiliser, axis=1))[0][0] 
        included_readout_syndrome[index] = s 
    if logger.isEnabledFor(logging.DEBUG): 
        try: 
            logger.debug("run: qubit_readout_error ('X' marks qubit measurement errors, not necessarily indicating a Pauli X\n{}".format(
                code.ascii_art(pauli=code.new_pauli(bsf=np.concatenate((qubit_readout_error, np.zeros(code.n_k_d[0], dtype=int))))))) 
            logger.debug("run: readout_syndrome (included into full code):\n{}".format(code.ascii_art(syndrome=included_readout_syndrome))) 
        except AttributeError: 
            logger.debug("run: qubit_readout_error={}".format(qubit_readout_error)) 
            logger.debug("run: included_readout_syndrome={}".format(included_readout_syndrome)) 
    # if logger.isEnabledFor(logging.DEBUG): 
    #     logger.debug(f"run: residual_error={residual_error}") 
    #     logger.debug(f"run: qubit_readout_error={qubit_readout_error}")
    #     logger.debug(f"run: stabiliser_readout_error={stabiliser_readout_error}") 
    #     logger.debug(f"run: readout_syndrome={readout_syndrome}") 
    #     logger.debug(f"run: correction={correction}") 

    # sanity checks 
    # commutes_with_stabilizers = np.all(pt.bsp(recovered, readout.conserved_stabilisers(code).T) == 0)
    # if not commutes_with_stabilizers:
    #     log_data = {
    #         # models
    #         'code': repr(code),
    #         'error_model': repr(error_model),
    #         'decoder': repr(decoder),
    #         # variables
    #         'error': pt.pack(error),
    #         'recovery': pt.pack(recovery),
    #         # step variables
    #         'step_errors': [pt.pack(v) for v in step_errors],
    #         'step_measurement_errors': [pt.pack(v) for v in step_measurement_errors],
    #     }
    #     logger.warning('RECOVERY DOES NOT RETURN TO (+1)-EIGENSPACE OF CONSERVED STABILIZERS: {}'.format(json.dumps(log_data, sort_keys=True)))
    #     logger.warning(str(code.ascii_art(pauli=code.new_pauli(bsf=residual_error)))) 
    #     logger.warning(str(code.ascii_art(pauli=code.new_pauli(bsf=correction))))     
    #     meas_err_as_qubit_err = np.concatenate([qubit_readout_error, [0]*code.n_k_d[0]])  
    #     logger.warning(str(code.ascii_art(pauli=code.new_pauli(bsf=meas_err_as_qubit_err)))) 
    #     total = meas_err_as_qubit_err ^ residual_error ^ correction 
    #     logger.warning(str(code.ascii_art(pauli=code.new_pauli(bsf=total)))) 


    commutes_with_logicals = pt.bsp(residual_error, readout.conserved_logicals(code).T) 
    conserved_logical_support = np.apply_along_axis(support, 1, readout.conserved_logicals(code)) 
    measurement_introduces_error = np.dot(qubit_readout_error, conserved_logical_support.T)%2 
    success = np.all((commutes_with_logicals + measurement_introduces_error + correction)%2 == 0) 
    # if logger.isEnabledFor(logging.DEBUG):
    #     logger.debug('run: commutes_with_stabilizers={}'.format(commutes_with_stabilizers))
    #     logger.debug('run: commutes_with_logicals={}'.format(commutes_with_logicals))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: success={}'.format(success))

    data = {
        #'error_weight': pt.bsf_wt(np.array(step_errors)),
        'success': success,
    }
    if results == "full_history":
        data["history"] = residual_error_history 

    return data

def run_ft(code, time_steps, num_cycles, error_model, decoder, readout, error_probability, measurement_error_probability, max_runs=None, max_failures=None, random_seed=None, results="simple", initial_error=None): 

    if max_runs is None and max_failures is None: 
        max_runs = 1 

    wall_time_start = time.perf_counter() 

    runs_data = {
        'code': code.label,
        'n_k_d': code.n_k_d,
        'time_steps': time_steps,
        'num_cycles': num_cycles, 
        'error_model': error_model.label,
        'decoder': decoder.label,
        'error_probability': error_probability,
        'measurement_error_probability': measurement_error_probability,
        'n_run': 0,
        'n_success': 0,
        'n_fail': 0,
        'error_weight_total': 0,
        'error_weight_pvar': 0.0,
        'logical_failure_rate': 0.0,
        'physical_error_rate': 0.0,
        'wall_time': 0.0,
        'seed': 0, 
        'version':qcext.__version__,
    }
    if results == "full_history": 
        runs_data["history"] = np.zeros((num_cycles+1, 2*code.n_k_d[0]), dtype=int)
    if initial_error is not None: 
        runs_data["initial_error"] = initial_error.tolist() 
    else: 
        runs_data["initial_error"] = None

    seed_sequence = np.random.SeedSequence(random_seed)
    runs_data['seed'] = seed_sequence.entropy 
    rng = np.random.default_rng(seed_sequence) 

    while ((max_runs is None or runs_data['n_run'] < max_runs)
       and (max_failures is None or runs_data['n_fail'] < max_failures)):
        # run simulation
        data = _run_once_ft(code, 1, num_cycles, error_model, decoder, readout, error_probability, measurement_error_probability,
                         rng, results=results, initial_error=initial_error)
        # increment run counts
        runs_data['n_run'] += 1
        if data['success']:
            runs_data['n_success'] += 1
        else:
            runs_data['n_fail'] += 1
        if results == "full_history": 
            runs_data["history"] += data["history"] 

    runs_data['wall_time'] = time.perf_counter() - wall_time_start 
    if results == "full_history": 
        runs_data["history"] = runs_data["history"].tolist() # for serialization with JSON 

    _add_rate_statistics(runs_data) 

    return runs_data 