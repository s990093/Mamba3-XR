import torch
import random

def generate_parity_data(batch_size, seq_len):
    """
    Parity Task
    Tests model's long-term precise discrete state memory & ability to avoid decay.
    """
    X = torch.randint(0, 2, (batch_size, seq_len))
    Y = torch.zeros_like(X)
    
    # Calculate cumulative XOR
    Y[:, 0] = X[:, 0]
    for t in range(1, seq_len):
        Y[:, t] = (Y[:, t-1] ^ X[:, t])

    # In Parity for sequence length generalization, we feed {0, 1} as floats and predict parity.
    return X.float().unsqueeze(-1), Y.long()

def generate_modular_arithmetic_data(batch_size, seq_len, with_brackets=False, modulo=5):
    """
    Modular Arithmetic Task (Chomsky Hierarchy)
    Tests model's state tracking through arithmetic operations.
    Vocabulary:
    0 to (modulo - 1) : Numbers
    modulo : '+'
    modulo + 1 : '-'
    modulo + 2 : '*'
    (if with_brackets)
    modulo + 3 : '('
    modulo + 4 : ')'
    """
    X = []
    Y = []
    
    op_plus = modulo
    op_minus = modulo + 1
    op_mul = modulo + 2
    op_open = modulo + 3
    op_close = modulo + 4
    
    vocab_size = modulo + 5 if with_brackets else modulo + 3

    for _ in range(batch_size):
        seq_x = []
        seq_y = []
        
        current_val = 0
        
        # Build the sequence token by token to calculate the running truth
        if not with_brackets:
            # Format: NUM OP NUM OP NUM ...
            num_tokens = seq_len // 2 + 1
            op_tokens = seq_len - num_tokens
            
            # Start with a number
            first_num = random.randint(0, modulo - 1)
            seq_x.append(first_num)
            seq_y.append(first_num % modulo)
            current_val = first_num
            
            for _ in range(op_tokens):
                op = random.choice([op_plus, op_minus, op_mul])
                num = random.randint(0, modulo - 1)
                
                seq_x.append(op)
                seq_y.append(-100) # Output ignored for operators
                
                seq_x.append(num)
                if op == op_plus:
                    current_val = (current_val + num) % modulo
                elif op == op_minus:
                    current_val = (current_val - num) % modulo
                elif op == op_mul:
                    current_val = (current_val * num) % modulo
                
                seq_y.append(current_val)
        else:
            # Format: loosely structured with brackets, e.g., ( NUM OP NUM ) OP NUM
            # For simplicity in this synthetic benchmark, we create valid random bracket structures.
            # To keep sequence generation tractable, we use a simple stack to evaluate running modulo.
            
            # Simplified generation for bounded seq_len
            tokens = []
            depth = 0
            # Just generate random valid arithmetic expressions
            # A more robust parser is needed for true random bracket expressions,
            # but for this benchmark we assemble predefined patterns repeatedly.
            pattern_len = 0
            while pattern_len < seq_len:
                if pattern_len == 0 or random.random() > 0.5:
                    # add "NUM OP " or "NUM "
                    if pattern_len == seq_len - 1:
                        tokens.append(random.randint(0, modulo - 1))
                        pattern_len += 1
                    else:
                        tokens.append(random.randint(0, modulo - 1))
                        tokens.append(random.choice([op_plus, op_minus, op_mul]))
                        pattern_len += 2
                else:
                    # add "( NUM OP NUM ) OP "
                    if pattern_len + 5 <= seq_len:
                        tokens.append(op_open)
                        tokens.append(random.randint(0, modulo - 1))
                        tokens.append(random.choice([op_plus, op_minus, op_mul]))
                        tokens.append(random.randint(0, modulo - 1))
                        tokens.append(op_close)
                        pattern_len += 5
                        if pattern_len < seq_len:
                            tokens.append(random.choice([op_plus, op_minus, op_mul]))
                            pattern_len += 1
                    else:
                        tokens.append(random.randint(0, modulo - 1))
                        pattern_len += 1
            
            # Trim to exact length
            tokens = tokens[:seq_len]
            # Ensure it ends with a number
            if tokens[-1] in [op_plus, op_minus, op_mul, op_open]:
                tokens[-1] = random.randint(0, modulo - 1)
            
            # Simple left-to-right evaluation ignoring strict PEMDAS for modular arithmetic tracking
            # evaluation left to right is standard for these synthetic language modeling tasks
            # unless specified to evaluate after full sequence. We evaluate running result.
            seq_x = tokens
            seq_y = [-100] * seq_len
            
            # To simplify, we'll just set the final token to predict the total evaluation
            # (In standard Chomsky MA tasks, they predict at the end)
            # We'll write a simple evaluator
            try:
                # Convert tokens to string expression and eval
                expr_str = ""
                for t in tokens:
                    if t < modulo: expr_str += str(t)
                    elif t == op_plus: expr_str += "+"
                    elif t == op_minus: expr_str += "-"
                    elif t == op_mul: expr_str += "*"
                    elif t == op_open: expr_str += "("
                    elif t == op_close: expr_str += ")"
                
                # Python eval might fail on malformed strings from simple truncation
                # Provide a fallback
                result = eval(expr_str) % modulo
                seq_y[-1] = result
            except:
                seq_y[-1] = 0 # Fallback
                
        X.append(torch.tensor(seq_x, dtype=torch.long))
        Y.append(torch.tensor(seq_y, dtype=torch.long))

    return torch.stack(X), torch.stack(Y)
