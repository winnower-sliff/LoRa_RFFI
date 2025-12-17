"""
FLOPs计算工具模块

该模块提供了计算深度学习模型浮点运算次数(FLOPs)和参数数量的功能，
用于评估模型的计算复杂度和大小。

依赖:
    thop: 用于模型分析的第三方库
"""

def calculate_flops_and_params(model, triplet_data):
    """
    计算给定模型的FLOPs和参数数量
    
    该函数使用thop库来分析模型的计算复杂度，支持普通模型和TripletNet模型。
    对于TripletNet模型，仅计算embedding_net部分的FLOPs。
    
    Args:
        model: 要分析的深度学习模型
        triplet_data: 三元组数据，格式为(anchor, positive, negative)
                     用于构建模型输入样本
        
    Returns:
        None: 结果直接打印到控制台
        
    注意:
        需要安装thop库: pip install thop
    """
    try:
        from thop import profile

        # 从三元组数据中提取单个样本作为输入
        dummy_a = triplet_data[0][0:1]
        dummy_p = triplet_data[1][0:1]
        dummy_n = triplet_data[2][0:1]

        # 针对TripletNet的特殊处理
        if hasattr(model, 'embedding_net'):
            print("TripletNet detected. Calculating FLOPs...")
            embedding_net = model.embedding_net
            # 只计算embedding_net部分的FLOPs
            flops_single, params = profile(embedding_net, inputs=(dummy_a,), verbose=False)
            total_flops = flops_single
            print(f"FLOPs: {total_flops / 1e6:.6f}M")
            print(f"Params: {params / 1e3:.2f}K")
        else:
            # 对普通模型，使用完整三元组输入进行计算
            total_flops, params = profile(model, inputs=(dummy_a, dummy_p, dummy_n), verbose=False)
            print(f"FLOPs: {total_flops / 1e6:.6f}M")
            print(f"Params: {params / 1e3:.2f}K")

        return total_flops, params
    except ImportError:
        print("THOP not installed. Please install it using 'pip install thop'")
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
