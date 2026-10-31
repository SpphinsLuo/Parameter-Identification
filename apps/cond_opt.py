import numpy as np
from scipy.optimize import minimize

from traj_opt.cond_aver import cond_aver
from traj_opt.constr import constr


def main():
    x0 = 0.1 * np.ones(42)  # 6*7
    lb = -0.5 * np.ones(42)
    up = 0.5 * np.ones(42)
    # options = optimoptions('fmincon', 'Display', 'iter');
    # The second optimoptions call replaces the first, so only MaxFunEvals remains.
    # options = optimoptions('fmincon','MaxFunEvals', 30000);
    # options = optimoptions('fmincon','TolFun',10^(-3));
    # options = optimoptions('fmincon','ObjectiveLimit',10^(-3));
    # options = optimoptions('fmincon','PlotFcns','optimplotfval');
    cond0 = cond_aver(x0)

    # fmincon nonlinear inequalities are c(x) <= 0.
    # SLSQP inequality constraints are fun(x) >= 0, so the sign is flipped.
    cons = {"type": "ineq", "fun": lambda x: -np.asarray(constr(x)[0], dtype=float)}
    result = minimize(
        cond_aver,
        x0,
        method="SLSQP",
        bounds=list(zip(lb, up)),
        constraints=cons,
        options={"maxiter": 30000, "disp": True},
    )
    x = result.x
    fval = result.fun
    # 显示结果
    print("===== 优化结果 =====")
    print("最优变量 x:")
    print(x)
    print()
    print("初始条件下的条件数 cond0:")
    print(cond0)
    print()
    print("优化后的条件数cond_min:")
    print(fval)
    print()


    return x, fval, cond0


if __name__ == "__main__":
    main()
