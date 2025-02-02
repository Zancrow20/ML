def read():
    return list(map(int, input().split()))

def solve():
    input_data = read()
    t_tests = int(input_data[0])
    pos = 1

    # Большое число для "бесконечного" верхнего предела
    INF = 10 ** 20

    outputs = []

    for _ in range(t_tests):
        n = int(read()[0])
        h = list(map(int, read()))
        a = list(map(int, read()))
        t_arr = list(map(int, read()))


        # Построим требуемый порядок p:
        # p[k] = индекс растения, которое должно оказаться на позиции k по высоте (k=0 -- самое высокое).
        p = [0] * n
        for i in range(n):
            p[t_arr[i]] = i

        # Будем хранить итоговые границы на D
        L = 0  # D >= L
        U = INF  # D < U

        possible = True

        # Проверяем подряд идущие растения в порядке p
        for k in range(n - 1):
            i = p[k]
            j = p[k + 1]
            dx = h[i] - h[j]
            dy = a[i] - a[j]

            if dy == 0:
                # dx + dy*D = dx > 0 ?
                if dx <= 0:
                    possible = False
                    break
                # Иначе ограничений нет
                continue

            if dy > 0:
                # dx + dy*D > 0 => dy*D > -dx => D > -dx/dy
                # Ищем нижнюю границу:
                M = -dx
                if M > 0:
                    # D > M/dy => D >= floor(M/dy) + 1
                    # целочисленное деление в Python = floor(M//dy) для M>0,dy>0
                    bound = (M // dy) + 1
                    if bound > L:
                        L = bound
                # если M <= 0, то это dx>=0 => уже при D=0 высота i не меньше высоты j,
                # так что никакого ужесточения снизу нет
            else:
                # dy < 0
                # dx + dy*D > 0 => D < -dx/dy
                # Пусть M = -dx, тогда D < M/dy. Нужно взять целые D.
                M = -dx
                # Обозначим X = M//|dy| -- это floor(M/|dy|),
                # но учитываем, что dy<0 => деление "M//dy" будет floor по отрицательным.
                # Точнее запишем сразу:
                #     D < (M)/(dy).
                # Если M % |dy| == 0, значит (M)/(dy) -- целое => D <= (то число - 1).
                # Иначе D <= floor( (M)/(dy) ).

                # Чтобы аккуратно сделать, используем обычное целочисл. деление:
                # bound_float = M / dy (вещественное), но обойдёмся без float:

                # "целое" = M//dy  (т.к. dy<0, это уже floor)
                bound_int = M // dy  # в Python это округление вниз (учитывая знак)

                # проверка на "ровно делится" (M % dy == 0) ?
                # но т.к. dy<0, остаток может быть отрицательный — лучше смотреть по абсолютному значению
                if M * dy < 0:
                    # (M>0, dy<0) или (M<0, dy<0), бывают разные случаи
                    # Но проще взять логику: если M % dy == 0 => делится нацело
                    if M % dy == 0:
                        # строго < bound_int => D <= bound_int - 1
                        up = bound_int - 1
                    else:
                        # D <= bound_int
                        up = bound_int
                else:
                    # M * dy >= 0 ??? тогда это dx <= 0 ? Случаи могут запутать.
                    # Но давайте будем последовательны: "D < M/dy"
                    # bound_int = floor(M/dy).
                    # если оно делится точно, тогда D <= bound_int - 1, иначе D <= bound_int
                    if M % dy == 0:
                        up = bound_int - 1
                    else:
                        up = bound_int

                # Теперь D <= up
                if up < U:
                    U = up

        # В конце у нас есть L и U, причём надо D >= 0
        if not possible:
            outputs.append("-1")
            continue

        # учтём D >= 0
        if L < 0:
            L = 0

        if L > U:
            # нет подходящего целого
            outputs.append("-1")
        else:
            # выбираем D = L (минимально возможный)
            # проверяем, что L <= U. Если L <= U, значит L подходит
            outputs.append(str(L))

    print("\n".join(outputs))

solve()