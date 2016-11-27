def field_size():
    minret = np.zeros([nT])
    maxret = np.zeros([nT])

    for tectal in range(nT):
        p = 0
        weight = 0
        while weight == 0:
            weight = Wpt[tectal, p]
            p += 1
        minret[tectal] = p-1

        p = nR-1
        weight = 0
        while weight == 0:
            weight = Wpt[tectal, p]
            p -= 1
        maxret[tectal] = p + 1

    rangeret = maxret - minret
    meanrange = np.mean(rangeret[:])

    return meanrange



def field_separation():
    fieldcentre = np.zeros([nT])

    for tectal in range(nT):
        total = 0
        weightsum = 0
        for p in range(nR):
            total += p*Wpt[tectal, p]
            weightsum += Wpt[tectal, p]
        fieldcentre[tectal] = total/weightsum

    totaldistance = 0
    for tectal in range(nT-1):
        totaldistance += abs(fieldcentre[tectal+1] - fieldcentre[tectal])
    meanseparation = totaldistance/(nT-1)

    return meanseparation